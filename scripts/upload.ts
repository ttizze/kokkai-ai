// scripts/upload.ts
import "dotenv/config";
import glob from "fast-glob";
import crypto from "node:crypto";
import { createReadStream, promises as fs } from "node:fs";
import path from "node:path";
import OpenAI from "openai";
import pLimit from "p-limit";

type Manifest = {
  vectorStoreId: string;
  files: Record<string, { fileId: string; sha256: string }>;
  byHash: Record<string, string>;
};

const GLOB = "kokkai/*.json";
const CONCURRENCY = Number(process.env.CONCURRENCY ?? 4);
const MAX_RETRIES = Number(process.env.MAX_RETRIES ?? 7);
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS ?? 1000 * 60);
const HEARTBEAT_SEC = Number(process.env.HEARTBEAT_SEC ?? 5);

const openai = new OpenAI({ maxRetries: 0, timeout: TIMEOUT_MS });

// -------- util: backoff/retry/logging --------
function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function isRetryableStatus(status?: number | null) {
  // 5xx / 429 / gateway系はリトライ
  return !!status && (status >= 500 || status === 429 || status === 408 || status === 425 || status === 409);
}

async function withRetry<T>(
  label: string,
  fn: () => Promise<T>,
  max = MAX_RETRIES
): Promise<T> {
  let attempt = 0;
  let lastErr: unknown;
  while (attempt < max) {
    try {
      if (attempt > 0) {
        console.warn(`[retry] ${label} attempt ${attempt + 1}/${max}`);
      }
      const res = await fn();
      return res;
    } catch (e: unknown) {
      lastErr = e;
      const { status, reqId, message: msg } = extractErrorDetails(e);
      const retryable = isRetryableStatus(status);
      console.warn(`[error] ${label} status=${status ?? "?"} req=${reqId ?? "-"} msg=${msg}`);
      if (!retryable || attempt === max - 1) break;
      const backoff = Math.min(30_000, Math.round((2 ** attempt) * 250 + Math.random() * 400));
      await sleep(backoff);
      attempt++;
    }
  }
  throw lastErr;
}

type MaybeResponseLike = {
  status?: number;
  response?: { status?: number };
  requestID?: string;
  headers?: { get?: (key: string) => string | null | undefined };
  error?: { message?: string };
  message?: string;
};

function extractErrorDetails(e: unknown): { status?: number | null; reqId?: string; message: string } {
  const err = (e ?? {}) as Partial<MaybeResponseLike>;
  const status = err.status ?? err.response?.status ?? null;
  const reqId = err.requestID ?? err.headers?.get?.("x-request-id") ?? undefined;
  const message = err.error?.message ?? err.message ?? String(e);
  return { status, reqId, message };
}

// -------- manifest I/O --------
async function sha256OfFile(filePath: string) {
  const buf = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(buf).digest("hex");
}

async function loadManifest(): Promise<Manifest | null> {
  try {
    const s = await fs.readFile("config/vectorStore.manifest.json", "utf-8");
    return JSON.parse(s) as Manifest;
  } catch {
    return null;
  }
}
async function saveManifest(m: Manifest) {
  await fs.mkdir("config", { recursive: true });
  await fs.writeFile("config/vectorStore.manifest.json", JSON.stringify(m, null, 2));
}

// -------- OpenAI helpers --------
async function ensureVectorStore(manifest: Manifest | null) {
  if (manifest?.vectorStoreId) return manifest.vectorStoreId;
  const vs = await withRetry("vectorStores.create", () =>
    openai.vectorStores.create({ name: "kokkai_jp" })
  );
  return vs.id;
}

async function listAttached(vectorStoreId: string): Promise<Set<string>> {
  const set = new Set<string>();
  let after: string | undefined;
  while (true) {
    const page = await withRetry("vectorStores.files.list", () =>
      openai.vectorStores.files.list(vectorStoreId, { limit: 100, after })
    );
    for (const f of page.data) set.add(f.id);
    if (!page.has_more) break;
    const last = page.data[page.data.length - 1];
    if (!last) break;
    after = last.id;
  }
  return set;
}

async function uploadWithRetry(filePath: string): Promise<string> {
  return await withRetry(`files.create ${path.basename(filePath)}`, async () => {
    const res = await openai.files.create({
      file: createReadStream(filePath),
      purpose: "assistants",
    });
    return res.id;
  });
}

async function attachIfMissing(vectorStoreId: string, fileId: string | undefined, already: Set<string>) {
  if (!fileId) throw new Error("fileId is undefined");
  if (already.has(fileId)) return;

  await withRetry("vectorStores.files.create", async () => {
    try {
      await openai.vectorStores.files.create(vectorStoreId, { file_id: fileId });
      already.add(fileId);
      return;
    } catch (e: unknown) {
      const { status } = extractErrorDetails(e);
      // 409: すでに添付済み（APIの仕様変更や並行時にあり得る）→成功扱い
      if (status === 409) {
        already.add(fileId);
        return;
      }
      throw e;
    }
  });
}

// -------- main --------
function ensureManifest(base: Manifest | null, vectorStoreId: string): Manifest {
  const m: Manifest = base ?? { vectorStoreId, files: {}, byHash: {} };
  if (m.vectorStoreId !== vectorStoreId) {
    m.vectorStoreId = vectorStoreId;
  }
  return m;
}

async function main() {
  const loaded = await loadManifest();
  const vectorStoreId = await ensureVectorStore(loaded);
  const manifest = ensureManifest(loaded, vectorStoreId);
  if (loaded?.vectorStoreId !== manifest.vectorStoreId) {
    await saveManifest(manifest);
  }

  const paths = await glob(GLOB, { absolute: true });
  if (!paths.length) {
    const manifestFileIds = Object.values(manifest.files ?? {}).map((f) => f.fileId);
    if (!manifestFileIds.length) {
      throw new Error(
        "no corpus files found. Place JSON files under 'kokkai/*.json' (note: '.gitignore' excludes '*kokkai*'), or provide a manifest with previously uploaded files."
      );
    }

    console.log(
      `[attach-only] no local files found; will (re)attach ${manifestFileIds.length} files to vectorStoreId=${vectorStoreId}`
    );
    const attached = await listAttached(vectorStoreId);
    const limit = pLimit(CONCURRENCY);

    let countDone = 0;
    let countAttachedNew = 0;
    let countSkipped = 0;
    let countFailed = 0;

    const tasks = manifestFileIds.map((fileId) =>
      limit(async () => {
        const beforeSize = attached.size;
        try {
          await attachIfMissing(vectorStoreId, fileId, attached);
          if (attached.size > beforeSize) countAttachedNew++;
          else countSkipped++;
        } catch (e: unknown) {
          countFailed++;
          const { status, reqId, message: msg } = extractErrorDetails(e);
          console.error(`[fail attach] ${fileId} status=${status ?? "?"} req=${reqId ?? "-"} msg=${msg}`);
        } finally {
          countDone++;
        }
      })
    );

    await Promise.allSettled(tasks);
    console.log(
      `[done attach-only] total=${manifestFileIds.length} attachedNew=${countAttachedNew} skipped=${countSkipped} failed=${countFailed}`
    );
    if (countFailed > 0) process.exit(2);
    console.log("vectorStoreId:", vectorStoreId);
    return;
  }

  console.log(`[start] files=${paths.length} vectorStoreId=${vectorStoreId}`);

  // 既存添付を取得（堅牢にリトライ）
  const attached = await listAttached(vectorStoreId);
  console.log(`[attached] already=${attached.size}`);

  const limit = pLimit(CONCURRENCY);

  // 進捗表示用カウンタ
  let countStarted = 0;
  let countDone = 0;
  let countUploaded = 0;
  let countAttachedOnly = 0;
  let countSkipped = 0;
  let countFailed = 0;

  const startedAt = Date.now();
  const hb = setInterval(() => {
    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    console.log(
      `[heartbeat ${elapsed}s] started=${countStarted}/${paths.length} done=${countDone} uploaded=${countUploaded} attachedOnly=${countAttachedOnly} skipped=${countSkipped} failed=${countFailed}`
    );
  }, HEARTBEAT_SEC * 1000);

  // Ctrl-C 時も manifest が落ちないように
  const onExit = async () => {
    clearInterval(hb);
    try {
      await saveManifest(manifest);
      console.log("[exit] manifest saved.");
    } catch (e: unknown) {
      console.error("[exit] failed to save manifest:", e);
    } finally {
      process.exit(1);
    }
  };
  process.on("SIGINT", onExit);
  process.on("SIGTERM", onExit);

  const tasks = paths.map((p) =>
    limit(async () => {
      countStarted++;
      const base = path.basename(p);
      try {
        // 既にこのローカルパスを処理済みならスキップ（再添付だけ確認）
        if (manifest.files[p]) {
          const { fileId } = manifest.files[p];
          console.log(`[resume] ${base} -> ${fileId}`);
          await attachIfMissing(vectorStoreId, fileId, attached);
          countSkipped++;
          countDone++;
          return;
        }

        console.log(`[hash] ${base}`);
        const hash = await sha256OfFile(p);
        const existingByHash = manifest.byHash[hash];
        if (existingByHash) {
          console.log(`[attach-only] ${base} -> ${existingByHash}`);
          await attachIfMissing(vectorStoreId, existingByHash, attached);
          manifest.files[p] = { fileId: existingByHash, sha256: hash };
          await saveManifest(manifest);
          countAttachedOnly++;
          countDone++;
          return;
        }

        console.log(`[upload] ${base}`);
        const fileId = await uploadWithRetry(p);
        console.log(`[attach] ${base} -> ${fileId}`);
        await attachIfMissing(vectorStoreId, fileId, attached);

        manifest.files[p] = { fileId, sha256: hash };
        manifest.byHash[hash] = fileId;
        await saveManifest(manifest);

        console.log(`[ok] ${base} -> ${fileId}`);
        countUploaded++;
        countDone++;
      } catch (e: unknown) {
        countFailed++;
        const { status, reqId, message: msg } = extractErrorDetails(e);
        console.error(`[fail] ${base} status=${status ?? "?"} req=${reqId ?? "-"} msg=${msg}`);
      }
    })
  );

  const results = await Promise.allSettled(tasks);
  clearInterval(hb);

  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  console.log(
    `[done ${elapsed}s] total=${paths.length} done=${countDone} uploaded=${countUploaded} attachedOnly=${countAttachedOnly} skipped=${countSkipped} failed=${countFailed}`
  );

  // 失敗があれば非ゼロ終了（CI 等で検知できる）
  const anyRejected = results.some((r) => r.status === "rejected");
  if (anyRejected || countFailed > 0) process.exit(2);

  console.log("vectorStoreId:", vectorStoreId);
}

main().catch((err) => {
  console.error("[fatal]", err);
  process.exit(1);
});
