#!/usr/bin/env python3
"""Simulated agent workload: multi-turn conversation with growing context."""
import aiohttp
import asyncio
import time
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "model"

SYSTEM_PROMPT = """You are a helpful AI assistant with expertise in software engineering, system design, and programming. You provide detailed, accurate responses with code examples when appropriate. You think step-by-step and explain your reasoning clearly."""

USER_TURNS = [
    "Explain the CAP theorem in distributed systems and give a real-world example of each trade-off.",
    "Now design a distributed key-value store that favors partition tolerance and availability (AP). Include the data model, replication strategy, and conflict resolution approach.",
    "Write a Python implementation of a gossip protocol for cluster membership and failure detection. Include heartbeats, suspicion mechanism, and cleanup logic.",
    "How would you add consistent hashing to this system? Show the ring implementation and virtual node mapping in Python.",
    "Design a monitoring system for this cluster that tracks request latency, error rates, and node health. Include alerting thresholds and escalation policies.",
    "Implement a circuit breaker pattern in Python that integrates with the key-value store client. Include states (closed/open/half-open), timeout logic, and fallback mechanisms.",
    "Write a chaos engineering test suite that validates the system's resilience to network partitions, node failures, and disk corruption.",
]

async def multi_turn_chat(session, num_turns=7, max_tokens_per_turn=200):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    all_tps = []
    all_ttft = []
    total_tokens = 0
    overall_start = time.perf_counter()
    
    for turn_idx in range(num_turns):
        messages.append({"role": "user", "content": USER_TURNS[turn_idx]})
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens_per_turn,
            "stream": True,
            "temperature": 0.7,
        }
        
        turn_start = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_response = ""
        
        async with session.post(API_URL, json=payload) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
        
        turn_end = time.perf_counter()
        ttft = (first_token_time - turn_start) * 1000 if first_token_time else 0
        gen_time = (turn_end - first_token_time) if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0
        
        all_ttft.append(ttft)
        all_tps.append(tps)
        total_tokens += token_count
        
        print(f"  Turn {turn_idx+1}: TTFT={ttft:.0f}ms, TPS={tps:.1f}, tokens={token_count}, "
              f"gen_time={gen_time:.2f}s, context_msgs={len(messages)}")
        
        # Add assistant response to context
        messages.append({"role": "assistant", "content": f"[Response {turn_idx+1}: {token_count} tokens]"})
    
    overall_time = time.perf_counter() - overall_start
    overall_tps = total_tokens / overall_time
    
    return {
        "turns": num_turns,
        "total_tokens": total_tokens,
        "overall_time_s": overall_time,
        "overall_tps": overall_tps,
        "per_turn_tps": all_tps,
        "per_turn_ttft": all_ttft,
    }

async def main():
    import statistics
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up...")
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20, "stream": True}
        async with session.post(API_URL, json=payload) as resp:
            async for line in resp.content:
                pass
        print()
        
        for max_tok in [200, 500]:
            print(f"\n{'='*60}")
            print(f"  AGENT SIMULATION: 7 turns, max_tokens={max_tok}/turn")
            print(f"{'='*60}")
            result = await multi_turn_chat(session, num_turns=7, max_tokens_per_turn=max_tok)
            
            print(f"\n  SUMMARY:")
            print(f"    Total tokens:  {result['total_tokens']}")
            print(f"    Total time:    {result['overall_time_s']:.1f}s")
            print(f"    Overall TPS:   {result['overall_tps']:.1f} tok/s")
            print(f"    Per-turn TPS:  min={min(result['per_turn_tps']):.1f}  "
                  f"avg={statistics.mean(result['per_turn_tps']):.1f}  "
                  f"max={max(result['per_turn_tps']):.1f}")
            print(f"    Per-turn TTFT: min={min(result['per_turn_ttft']):.0f}ms  "
                  f"avg={statistics.mean(result['per_turn_ttft']):.0f}ms  "
                  f"max={max(result['per_turn_ttft']):.0f}ms")
            
            # Check for degradation
            first_half = statistics.mean(result['per_turn_tps'][:3])
            second_half = statistics.mean(result['per_turn_tps'][3:])
            if second_half < first_half * 0.8:
                print(f"    *** DEGRADATION: TPS dropped {(1-second_half/first_half)*100:.0f}% from first half to second half ***")
            else:
                print(f"    No significant degradation (1st half: {first_half:.1f}, 2nd half: {second_half:.1f})")

if __name__ == "__main__":
    asyncio.run(main())
