#!/usr/bin/env python3
"""Single-stream benchmark to measure isolated request performance (agent workload)."""
import aiohttp
import asyncio
import time
import statistics
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "model"

# 3000-char prompt (same as vllm_bench.py)
PROMPT_3K = """You are a Principal Systems Engineer tasked with designing and implementing a high-performance, distributed task scheduling system. The system must be written in modern C++ (C++20) and use PostgreSQL as its persistent state store and sole coordination mechanism. This scheduler will be deployed across a cluster of identical worker nodes and must be capable of handling up to 10,000 task state transitions per second without introducing lock contention or deadlocks in the database.

Your response must thoroughly address the following architectural and implementation requirements:

PostgreSQL Schema and Concurrency Control:
Design the database schema to store tasks. Each task has an ID, a JSON payload, a priority level, a state (Pending, Processing, Completed, Failed), a creation timestamp, and a visibility timeout.
Provide the exact SQL queries required for a worker node to claim a batch of 'Pending' tasks. You must utilize PostgreSQL's FOR UPDATE SKIP LOCKED mechanism to ensure multiple concurrent C++ worker nodes do not attempt to claim the same tasks. Explain your indexing strategy in detail, particularly how you will prevent index bloat on heavily updated tables and optimize for the 'find highest priority pending task' query.

C++ Worker Node Architecture:
Outline the internal architecture of a single C++ worker node. The node must fetch tasks from the database, execute them asynchronously, and update their states.
Detail your threading model. Will you use a thread pool, an actor model, or asynchronous I/O (e.g., io_uring or Boost.Asio)? Justify your choice based on the requirement to handle high throughput with minimal CPU context-switching overhead.
Provide a core C++ code snippet demonstrating the thread-safe worker loop that fetches tasks, dispatches them to a processing pool, and handles the completion callbacks. Use modern C++ synchronization primitives where appropriate, but explain how you minimize lock contention within the worker process itself.

Fault Tolerance and Edge Cases:
Describe how the system handles worker node crashes. If a C++ process dies while holding a claimed task (state = 'Processing'), how is that task recovered and reassigned? Implement a visibility timeout mechanism and provide the SQL required to sweep and reset orphaned tasks back to the 'Pending' queue.
Address the 'thundering herd' problem. If the PostgreSQL database briefly goes offline and comes back, how do you prevent all active C++ worker nodes from simultaneously hammering the database with connection requests and polling queries? Outline a jittered exponential backoff strategy in your C++ code design.

Performance Optimization:
Discuss how you would monitor and minimize database transaction latency. What specific PostgreSQL configuration parameters (e.g., shared_buffers, work_mem, effective_io_concurrency) would you tune for this specific read-modify-write heavy workload?
In your C++ implementation, how do you manage memory allocation to avoid fragmentation during sustained high-load operations? Consider the use of custom allocators or memory pools for handling the JSON task payloads.

Deliverables:

Complete SQL schema definitions and the critical transactional queries.

C++20 class definitions and the core execution loop implementation.

A brief architectural diagram (described in text) of the system components.

A structured explanation of your design trade-offs regarding concurrency and fault recovery.

Constraint: Do not use any external message brokers or caching layers like RabbitMQ, Kafka, or Redis. The entire coordination mechanism must rely exclusively on PostgreSQL and the C++ application logic."""

# Short prompt for comparison
PROMPT_SHORT = "What is 2+2? Explain briefly."

async def single_request(session, prompt, max_tokens, stream=True):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async with session.post(API_URL, json=payload) as response:
        if stream:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
        else:
            data = await response.json()
            usage = data.get("usage", {})
            token_count = usage.get("completion_tokens", 0)
            first_token_time = time.perf_counter()  # no per-token timing
    
    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    total_time = end - start
    gen_time = (end - first_token_time) if first_token_time else 0
    tps = token_count / gen_time if gen_time > 0 else 0
    
    return {
        "ttft_ms": ttft,
        "total_s": total_time,
        "gen_s": gen_time,
        "tps": tps,
        "tokens": token_count,
    }

async def main():
    tests = [
        ("short_100tok", PROMPT_SHORT, 100),
        ("3K_100tok", PROMPT_3K, 100),
        ("3K_500tok", PROMPT_3K, 500),
        ("3K_1000tok", PROMPT_3K, 1000),
    ]
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up (3 requests)...")
        for _ in range(3):
            await single_request(session, "Hello", 20)
        print()
        
        results = {}
        for label, prompt, max_tok in tests:
            print(f"Running: {label} (max_tokens={max_tok})")
            run_results = []
            # 3 runs each, sequential
            for i in range(3):
                r = await single_request(session, prompt, max_tok)
                run_results.append(r)
                print(f"  Run {i+1}: TTFT={r['ttft_ms']:.0f}ms, TPS={r['tps']:.1f}, tokens={r['tokens']}, total={r['total_s']:.2f}s")
            
            results[label] = run_results
        
        print("\n" + "=" * 70)
        print("  SINGLE-STREAM BENCHMARK RESULTS")
        print("=" * 70)
        for label, runs in results.items():
            ttfts = [r['ttft_ms'] for r in runs]
            tps_vals = [r['tps'] for r in runs]
            tokens = [r['tokens'] for r in runs]
            print(f"\n  {label}:")
            print(f"    TTFT:   avg={statistics.mean(ttfts):.0f}ms  median={statistics.median(ttfts):.0f}ms")
            print(f"    TPS:    avg={statistics.mean(tps_vals):.1f}  median={statistics.median(tps_vals):.1f}")
            print(f"    Tokens: avg={statistics.mean(tokens):.0f}")

if __name__ == "__main__":
    asyncio.run(main())
