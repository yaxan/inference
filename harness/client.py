#!/usr/bin/env python3
import argparse, asyncio, json, time, uuid, pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

def load_prompts(path):
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            items.append((obj["name"], obj["prompt"]))
    return items

async def run_one(client, model, prompt, max_tokens, temperature, stream, timeout_s):
    start = time.perf_counter()
    ttft = None
    completion_tokens = None

    if stream:
        # Streaming to measure TTFT
        st = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            timeout=timeout_s,
        )
        first = True
        text = []
        for ev in st:
            if first:
                ttft = time.perf_counter() - start
                first = False
            delta = ev.choices[0].delta.content or ""
            text.append(delta)
        e2e = time.perf_counter() - start
        # We don't know tokens from streaming; mark as None
        return {"e2e_s": e2e, "ttft_s": ttft, "completion_tokens": None, "text_len": sum(len(t) for t in text)}
    else:
        # Non-streaming → get token counts
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            timeout=timeout_s,
        )
        e2e = time.perf_counter() - start
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        return {"e2e_s": e2e, "ttft_s": None, "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens, "total_tokens": total_tokens}

async def run_sweep(base_url, api_key, model, prompts_path, out_csv, concurrency, runs_per_prompt,
                    max_tokens, temperature, stream, timeout_s):
    client = OpenAI(base_url=base_url, api_key=api_key)
    prompts = load_prompts(prompts_path)

    rows = []
    sem = asyncio.Semaphore(concurrency)

    async def wrap(name, prompt, run_id):
        async with sem:
            res = await run_one(client, model, prompt, max_tokens, temperature, stream, timeout_s)
            res.update({"prompt_name": name, "run_id": run_id, "concurrency": concurrency,
                        "max_tokens": max_tokens, "temperature": temperature, "stream": stream})
            rows.append(res)

    tasks = []
    for name, prompt in prompts:
        for r in range(runs_per_prompt):
            tasks.append(asyncio.create_task(wrap(name, prompt, r)))

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompts", default="../prompts/golden.jsonl")
    ap.add_argument("--out", default="../reports/baseline.csv")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--runs-per-prompt", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=120)
    args = ap.parse_args()

    asyncio.run(run_sweep(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        prompts_path=args.prompts, out_csv=args.out, concurrency=args.concurrency,
        runs_per_prompt=args.runs_per_prompt, max_tokens=args.max_tokens,
        temperature=args.temperature, stream=args.stream, timeout_s=args.timeout_s,
    ))

if __name__ == "__main__":
    main()
