#!/usr/bin/env python3
"""Test with very long system prompt to stress KV cache and check for degradation."""
import aiohttp
import asyncio
import time
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "model"

# Very long system prompt (~8000 chars = ~2000-3000 tokens)
LONG_SYSTEM = """You are an expert AI coding assistant with deep knowledge of multiple programming languages, frameworks, and system design patterns. You follow these principles:

1. CODE QUALITY: Write clean, well-documented code with proper error handling. Use type hints in Python, proper const correctness in C++, and idiomatic patterns in each language.

2. ARCHITECTURE: When designing systems, consider scalability, reliability, and maintainability. Document trade-offs explicitly. Use established patterns (CQRS, Event Sourcing, Saga, etc.) where appropriate.

3. SECURITY: Always consider security implications. Validate inputs, use parameterized queries, implement proper authentication and authorization, and follow the principle of least privilege.

4. PERFORMANCE: Profile before optimizing. Consider algorithmic complexity, memory usage, and I/O patterns. Use appropriate data structures and avoid premature optimization.

5. TESTING: Write comprehensive tests including unit tests, integration tests, and edge cases. Use test-driven development when appropriate. Aim for high code coverage.

6. DOCUMENTATION: Document public APIs, architectural decisions, and non-obvious code. Use inline comments for complex logic and docstrings for functions and classes.

7. DEVOPS: Consider deployment, monitoring, logging, and alerting. Implement health checks, graceful shutdown, and circuit breakers. Use infrastructure as code.

8. DATABASE: Design schemas carefully. Consider indexing strategies, query performance, migration paths, and data integrity. Use appropriate isolation levels and handle concurrent access correctly.

9. CONCURRENCY: Design for concurrent access. Use appropriate synchronization primitives. Avoid deadlocks and race conditions. Consider both thread-based and async models.

10. API DESIGN: Follow REST conventions or use GraphQL where appropriate. Version your APIs. Implement rate limiting, pagination, and proper HTTP status codes.

When responding to coding questions:
- Start with a brief overview of your approach
- Provide working code with comments
- Explain key design decisions
- Mention potential improvements or alternatives
- Consider error cases and edge conditions

When debugging:
- Ask clarifying questions first
- Reproduce the issue systematically
- Formulate hypotheses before testing
- Verify fixes don't introduce regressions

When reviewing code:
- Check for correctness first
- Then look for performance issues
- Then check for security vulnerabilities
- Finally suggest style improvements

You have access to common tools: file read/write, shell commands, web search, and code execution. Use them judiciously to provide accurate, tested answers.

Always think step by step, especially for complex problems. Break large tasks into smaller sub-tasks and address them one at a time.

Format code blocks with appropriate language tags. Use markdown for structured responses. Keep responses focused and actionable.

Remember: a working simple solution is better than a perfect complex one. Ship early, iterate often, and measure everything."""

SHORT_QUESTIONS = [
    "What's the difference between a mutex and a semaphore in C++?",
    "How does garbage collection work in Python vs Go?",
    "Explain the actor model of concurrency.",
    "What are the trade-offs between SQL and NoSQL databases?",
    "How does Kubernetes handle pod scheduling and scaling?",
    "Explain the CQRS pattern with a concrete example.",
    "What's the difference between HTTP/2 and HTTP/3?",
    "How would you design a rate limiter for an API?",
    "Explain eventual consistency with a real-world example.",
    "What are the benefits and drawbacks of microservices vs monoliths?",
    "How does distributed tracing work across microservices?",
    "Explain the CAP theorem and give examples of each trade-off.",
    "What is backpressure in reactive systems and how do you implement it?",
    "How do you handle schema evolution in a database?",
    "What are the best practices for API versioning?",
]

async def long_context_bench(session):
    # Test with the long system prompt and many short turns
    messages = [{"role": "system", "content": LONG_SYSTEM}]
    
    print(f"System prompt: ~{len(LONG_SYSTEM)} chars")
    print()
    
    per_turn = []
    for turn_idx in range(len(SHORT_QUESTIONS)):
        messages.append({"role": "user", "content": SHORT_QUESTIONS[turn_idx]})
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 150,
            "stream": True,
            "temperature": 0.7,
        }
        
        start = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        async with session.post(API_URL, json=payload) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
        
        end = time.perf_counter()
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        gen_time = (end - first_token_time) if first_token_time else 0
        tps = token_count / gen_time if gen_time > 0 else 0
        
        per_turn.append({"ttft": ttft, "tps": tps, "tokens": token_count, "msgs": len(messages)})
        print(f"  Turn {turn_idx+1:2d}: TTFT={ttft:6.0f}ms  TPS={tps:5.1f}  tokens={token_count:4d}  context_msgs={len(messages):2d}")
        
        messages.append({"role": "assistant", "content": f"[Response: {token_count} tokens]"})
    
    return per_turn

async def main():
    import statistics
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up...")
        for _ in range(3):
            payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20, "stream": True}
            async with session.post(API_URL, json=payload) as resp:
                async for line in resp.content:
                    pass
        print()
        
        print("=" * 70)
        print("  LONG CONTEXT AGENT SIMULATION")
        print("  Long system prompt + 15 short turns, 150 tok/turn")
        print("=" * 70)
        
        results = await long_context_bench(session)
        
        tps_vals = [r['tps'] for r in results]
        ttft_vals = [r['ttft'] for r in results]
        
        print(f"\n  SUMMARY:")
        print(f"    TPS:  min={min(tps_vals):.1f}  avg={statistics.mean(tps_vals):.1f}  max={max(tps_vals):.1f}")
        print(f"    TTFT: min={min(ttft_vals):.0f}ms  avg={statistics.mean(ttft_vals):.0f}ms  max={max(ttft_vals):.0f}ms")
        
        # Check degradation
        first_5 = statistics.mean(tps_vals[:5])
        last_5 = statistics.mean(tps_vals[-5:])
        if last_5 < first_5 * 0.8:
            print(f"    *** DEGRADATION: TPS dropped {(1-last_5/first_5)*100:.0f}% (first 5: {first_5:.1f}, last 5: {last_5:.1f}) ***")
        else:
            print(f"    Minimal degradation (first 5: {first_5:.1f}, last 5: {last_5:.1f})")

if __name__ == "__main__":
    asyncio.run(main())
