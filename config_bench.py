#!/usr/bin/env python3
"""Config benchmark: test different KV cache / SDMA / context settings.

Runs a standardized agent simulation workload and measures:
- TTFT (time to first token)
- Generation throughput (tok/s)
- KV cache utilization
- Stability (no crashes across all turns)

Usage:
  python3 config_bench.py [--turns 7] [--max-tokens 200] [--label "my test"]
"""
import json
import sys
import time
import urllib.request
import argparse


API_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"

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


def get_metrics():
    """Get key vLLM metrics."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as r:
            text = r.read().decode()
        metrics = {}
        for line in text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                name = parts[0].split('{')[0].replace('vllm:', '')
                try:
                    metrics[name] = float(parts[1])
                except ValueError:
                    pass
        return metrics
    except Exception as e:
        return {"error": str(e)}


def get_gen_tokens():
    metrics = get_metrics()
    return metrics.get('generation_tokens_total', 0)


def make_request(messages, max_tokens=200, temperature=0.7):
    """Make a single chat completion request. Returns (response_dict, elapsed)."""
    payload = json.dumps({
        "model": "model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_agent_sim(num_turns=7, max_tokens=200, label="test"):
    """Run a multi-turn agent simulation and report metrics."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    turn_results = []
    overall_start = time.perf_counter()

    for i in range(min(num_turns, len(USER_TURNS))):
        messages.append({"role": "user", "content": USER_TURNS[i]})

        # Measure TTFT with streaming-like approach
        gen_before = get_gen_tokens()
        result, elapsed = make_request(messages, max_tokens=max_tokens)

        gen_after = get_gen_tokens()
        comp_tokens = int(gen_after - gen_before) if gen_after > gen_before else 0

        # Also get from response
        resp_tokens = result.get('usage', {}).get('completion_tokens', comp_tokens)
        prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)

        tps = resp_tokens / elapsed if elapsed > 0 else 0

        turn_results.append({
            'turn': i + 1,
            'prompt_tokens': prompt_tokens,
            'comp_tokens': resp_tokens,
            'elapsed': elapsed,
            'tps': tps,
            'context_msgs': len(messages),
        })

        # Add assistant response to context
        content = result['choices'][0]['message'].get('content', '') or ''
        reasoning = result['choices'][0]['message'].get('reasoning', '') or ''
        assistant_text = content if content else reasoning[:500]
        messages.append({"role": "assistant", "content": assistant_text})

    total_time = time.perf_counter() - overall_start
    total_tokens = sum(t['comp_tokens'] for t in turn_results)
    overall_tps = total_tokens / total_time if total_time > 0 else 0

    tps_values = [t['tps'] for t in turn_results]

    # Get final KV cache metrics
    final_metrics = get_metrics()

    # Print results
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"  {num_turns} turns, max_tokens={max_tokens}/turn")
    print(f"{'='*60}")

    for t in turn_results:
        print(f"  Turn {t['turn']}: TPS={t['tps']:.1f}, tokens={t['comp_tokens']}, "
              f"time={t['elapsed']:.2f}s, ctx_msgs={t['context_msgs']}, "
              f"prompt_tokens={t['prompt_tokens']}")

    half = len(tps_values) // 2
    first_half = sum(tps_values[:half]) / half if half > 0 else 0
    second_half = sum(tps_values[half:]) / (len(tps_values) - half) if len(tps_values) > half else 0

    degradation = "Significant" if first_half > 0 and (second_half / first_half) < 0.7 else "Minimal"

    print(f"\n  SUMMARY:")
    print(f"    Total tokens:  {total_tokens}")
    print(f"    Total time:    {total_time:.1f}s")
    print(f"    Overall TPS:   {overall_tps:.1f} tok/s")
    print(f"    Per-turn TPS:  min={min(tps_values):.1f}  avg={sum(tps_values)/len(tps_values):.1f}  max={max(tps_values):.1f}")
    print(f"    Degradation:   {degradation} (1st half: {first_half:.1f}, 2nd half: {second_half:.1f})")
    print(f"    KV cache:      {final_metrics.get('kv_cache_usage_perc', 'N/A')}")
    print(f"    Max concurrency: {final_metrics.get('max_num_seqs', 'N/A')}")

    return {
        'label': label,
        'overall_tps': overall_tps,
        'min_tps': min(tps_values),
        'avg_tps': sum(tps_values) / len(tps_values),
        'max_tps': max(tps_values),
        'degradation': degradation,
        'total_tokens': total_tokens,
        'total_time': total_time,
    }


def run_single_stream_test(max_tokens=500, label="test"):
    """Run a single long-generation request for pure decode throughput."""
    gen_before = get_gen_tokens()

    payload = json.dumps({
        "model": "model",
        "messages": [{"role": "user", "content": "Write a detailed explanation of how transformer attention works, covering multi-head attention, positional encoding, and the softmax scaling factor."}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - start

    comp_tokens = result.get('usage', {}).get('completion_tokens', 0)
    tps = comp_tokens / elapsed if elapsed > 0 else 0

    print(f"  Single-stream ({label}): {comp_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    return {'label': f'single_{label}', 'tps': tps, 'tokens': comp_tokens, 'time': elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--single-tokens", type=int, default=500)
    parser.add_argument("--label", type=str, default="unnamed")
    parser.add_argument("--single-only", action="store_true")
    args = parser.parse_args()

    print(f"Warming up...")
    make_request([{"role": "user", "content": "Hello"}], max_tokens=10)

    if args.single_only:
        run_single_stream_test(max_tokens=args.single_tokens, label=args.label)
    else:
        result = run_agent_sim(num_turns=args.turns, max_tokens=args.max_tokens, label=args.label)
        print()
        run_single_stream_test(max_tokens=args.single_tokens, label=args.label)
