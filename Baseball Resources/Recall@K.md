recall@1 = 0.750 means **15 of your 20 in-coverage questions** had a correct source at rank 1 — not "1 correct." The `@k` is the cutoff position, not a count.

Take IC-01 "How do I hit the outside pitch?" Retrieval returns 5 chunks. If the outside-pitch video is 1st, that question counts for recall@1, @3, and @5. If it's 4th, it counts only for recall@5 — good enough for the LLM (it sees all 5), but ranked poorly.

Your numbers decompose like this:

|Metric|Value|Meaning|
|---|---|---|
|recall@1|0.750|15/20 correct at position 1|
|recall@3|0.750|still 15/20 — **nothing new appeared at positions 2–3**|
|recall@5|0.850|17/20 — two more surfaced at positions 4–5|
|MRR|0.775|average of 1/rank; high because most hits are at rank 1|

That flat @1→@3 is informative: your retrieval either nails it immediately or misses badly. Nothing lands mid-pack.

recall@5 = 0.85 already hits your target. The remaining 3 are the listed misses.