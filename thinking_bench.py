#!/usr/bin/env python3
"""Measure actual engine TPS vs perceived client TPS, accounting for reasoning tokens."""
import aiohttp
import asyncio
import time
import json

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "model"

async def get_metrics(session):
    async with session.get("http://localhost:8000/metrics") as resp:
        text = await resp.text()
    gen_tokens = 0
    for line in text.split('\n'):
        if line.startswith('vllm:generation_tokens_total{'):
            gen_tokens = float(line.split()[-1])
    return gen_tokens

async def bench(session, msg, max_tokens, label):
    # Get engine token count before
    gen_before = await get_metrics(session)
    
    start = time.perf_counter()
    async with session.post(API_URL, json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
    }) as resp:
        result = await resp.json()
    elapsed = time.perf_counter() - start
    
    # Get engine token count after
    gen_after = await get_metrics(session)
    
    usage = result.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    # Engine actually generated
    engine_tokens = int(gen_after - gen_before)
    
    # Reasoning tokens
    reasoning = result['choices'][0]['message'].get('reasoning', '') or ''
    reasoning_tokens_approx = len(reasoning) // 4  # rough estimate
    
    content = result['choices'][0]['message'].get('content', '') or ''
    
    engine_tps = engine_tokens / elapsed if elapsed > 0 else 0
    client_tps = completion_tokens / elapsed if elapsed > 0 else 0
    
    print(f"  {label}")
    print(f"    Engine TPS:        {engine_tps:6.1f} tok/s ({engine_tokens} tokens in {elapsed:.2f}s)")
    print(f"    Reported TPS:      {client_tps:6.1f} tok/s ({completion_tokens} tokens)")
    print(f"    Reasoning chars:   {len(reasoning)} (~{reasoning_tokens_approx} tokens)")
    print(f"    Content:           {repr(content[:100])}")
    print()

async def main():
    async with aiohttp.ClientSession() as session:
        # Test 1: Simple question, low max_tokens
        await bench(session, "What is 2+2?", 50, "simple_50tok")
        
        # Test 2: Simple question, more headroom
        await bench(session, "What is 2+2?", 500, "simple_500tok")
        
        # Test 3: Non-thinking request (disable thinking)
        start = time.perf_counter()
        gen_before = await get_metrics(session)
        async with session.post(API_URL, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 500,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }) as resp:
            result = await resp.json()
        elapsed = time.perf_counter() - start
        gen_after = await get_metrics(session)
        engine_tokens = int(gen_after - gen_before)
        completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
        content = result['choices'][0]['message'].get('content', '') or ''
        reasoning = result['choices'][0]['message'].get('reasoning', '') or ''
        engine_tps = engine_tokens / elapsed if elapsed > 0 else 0
        client_tps = completion_tokens / elapsed if elapsed > 0 else 0
        print(f"  no_thinking_500tok")
        print(f"    Engine TPS:        {engine_tps:6.1f} tok/s ({engine_tokens} tokens in {elapsed:.2f}s)")
        print(f"    Reported TPS:      {client_tps:6.1f} tok/s ({completion_tokens} tokens)")
        print(f"    Reasoning chars:   {len(reasoning)}")
        print(f"    Content:           {repr(content[:200])}")
        print()
        
        # Test 4: /think off in message
        await bench(session, "/no_think\nWhat is 2+2?", 500, "no_think_tag_500tok")

if __name__ == "__main__":
    asyncio.run(main())
