import type { NextRequest } from "next/server";
import OpenAI from "openai";
const vectorStoreId = process.env.VECTOR_STORE_ID || "";
const openai = new OpenAI();

export async function POST(req: NextRequest) {
	const { messages } = await req.json();
	const res = await openai.responses.create({
		model: "o3-mini",
		tools: [{ type: "file_search", vector_store_ids: [vectorStoreId], max_num_results: 10 }],
		include: ["file_search_call.results"],
		input: [
			{
				type: "message",
				role: "system",
				content: `あなたは国会議員が質問する前に､国会議事録を検索するためのAIです｡
				あなたには国会議事録が与えられています｡
				あなたは議事録を検索して､議員の質問に対する回答を生成してください｡
				`,
			},
			...messages.map((m: { role: string; content: string }) => ({
				type: "message",
				role: m.role,
				content: m.content,
			})),
		],
	});
	const citationBlocks: string[] = [];

	for (const item of res.output ?? []) {
		if (item.type === "file_search_call" && item.results) {
			for (const r of item.results) {
				citationBlocks.push(
					`- **${r.filename ?? r.file_id}**\n  > ${r.text?.trim()}`,
				);
			}
		}
	}

	const answer = citationBlocks.length
		? `${res.output_text.trim()}

---

<details>
<summary>📚 出典</summary>

${citationBlocks.join("\n\n")}

</details>`
		: res.output_text;
	/* ──────────────────────────────────────── */

	return new Response(answer, {
		headers: { "Content-Type": "text/markdown; charset=utf-8" },
	});
}
