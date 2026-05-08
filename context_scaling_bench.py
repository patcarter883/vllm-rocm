#!/usr/bin/env python3
"""Test performance vs context length to find the degradation point."""
import time
import json
import urllib.request

API_URL = "http://localhost:8000/v1/chat/completions"

def get_gen_tokens():
    with urllib.request.urlopen("http://localhost:8000/metrics") as r:
        text = r.read().decode()
    for line in text.split('\n'):
        if line.startswith('vllm:generation_tokens_total{'):
            return float(line.split()[-1])
    return 0

def bench(system_prompt_len, user_msg, max_tokens, label):
    gen_before = get_gen_tokens()
    
    system = "X " * system_prompt_len if system_prompt_len > 0 else ""
    payload = json.dumps({
        "model": "model",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
    }).encode()
    
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    
    gen_after = get_gen_tokens()
    engine_tokens = int(gen_after - gen_before)
    engine_tps = engine_tokens / elapsed if elapsed > 0 else 0
    
    prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
    comp_tokens = result.get('usage', {}).get('completion_tokens', 0)
    
    print(f"{label:25s}  prompt={prompt_tokens:5d}  gen={engine_tokens:4d}  engine_tps={engine_tps:5.1f}  elapsed={elapsed:.2f}s")
    return engine_tps

print("=== Context length vs decode speed ===")
print()

# No system prompt
bench(0, "Say hello.", 100, "0_tokens_ctx")

# ~250 tokens
bench(125, "Say hello.", 100, "250_tokens_ctx")

# ~500 tokens
bench(250, "Say hello.", 100, "500_tokens_ctx")

# ~1000 tokens
bench(500, "Say hello.", 100, "1K_tokens_ctx")

# ~2000 tokens
bench(1000, "Say hello.", 100, "2K_tokens_ctx")

# ~4000 tokens
bench(2000, "Say hello.", 100, "4K_tokens_ctx")

# ~8000 tokens
bench(4000, "Say hello.", 100, "8K_tokens_ctx")

# ~16000 tokens (roughly what OpenCode sends)
bench(8000, "Say hello.", 100, "16K_tokens_ctx")

# ~32000 tokens
bench(16000, "Say hello.", 100, "32K_tokens_ctx")
