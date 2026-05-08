#!/usr/bin/env python3
"""Clean single-request benchmark with precise timing."""
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

def bench(msg, max_tokens, label):
    gen_before = get_gen_tokens()
    
    payload = json.dumps({
        "model": "model",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
    }).encode()
    
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    
    gen_after = get_gen_tokens()
    engine_tokens = int(gen_after - gen_before)
    engine_tps = engine_tokens / elapsed if elapsed > 0 else 0
    
    usage = result.get('usage', {})
    comp_tokens = usage.get('completion_tokens', 0)
    client_tps = comp_tokens / elapsed if elapsed > 0 else 0
    
    reasoning = result['choices'][0]['message'].get('reasoning', '') or ''
    content = result['choices'][0]['message'].get('content', '') or ''
    
    print(f"{label}")
    print(f"  Engine:   {engine_tps:6.1f} tok/s  ({engine_tokens} tokens in {elapsed:.2f}s)")
    print(f"  Client:   {client_tps:6.1f} tok/s  ({comp_tokens} tokens)")
    print(f"  Thinking: {len(reasoning)} chars")
    print(f"  Content:  {repr(content[:120])}")
    print()
    return engine_tps

print("=== Fresh benchmark ===")
print()

# Simple question, small output
bench("Say hello.", 50, "hello_50")

# Simple question, larger output
bench("Explain what a variable is in programming in 2 sentences.", 200, "variable_200")

# Request WITHOUT thinking - use /no_think prefix
bench("/no_think\nSay hello.", 50, "no_think_hello_50")

# Request WITHOUT thinking, more output
bench("/no_think\nExplain what a variable is in programming in 2 sentences.", 200, "no_think_variable_200")
