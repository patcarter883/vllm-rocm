#!/usr/bin/env python3
"""Find max model length for a given KV cache dtype.

Tests progressively longer context lengths, checking:
1. Can the server start without OOM?
2. What's the concurrency level?
3. Performance at each length

Usage:
  python3 find_max_ctx.py --kv-dtype int8_per_token_head --min-concurrency 4
"""
import subprocess
import time
import re
import sys
import os

COMPOSE_DIR = "/home/pat/code/vllm-rocm"
COMPOSE_FILE = os.path.join(COMPOSE_DIR, "docker-compose.yml")


def run_cmd(cmd, timeout=300):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=COMPOSE_DIR)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def patch_config(kv_dtype, max_model_len):
    with open(COMPOSE_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'--kv-cache-dtype\s+\S+', f'--kv-cache-dtype {kv_dtype}', content)
    content = re.sub(r'--max-model-len\s+\S+', f'--max-model-len {max_model_len}', content)
    with open(COMPOSE_FILE, 'w') as f:
        f.write(content)


def restart():
    run_cmd("docker compose --profile qwen down", timeout=30)
    time.sleep(2)
    rc, _, err = run_cmd("docker compose --profile qwen up -d --force-recreate", timeout=60)
    return rc == 0


def wait_healthy(max_wait=180):
    start = time.time()
    while time.time() - start < max_wait:
        rc, _, _ = run_cmd("curl -sf http://localhost:8000/health", timeout=10)
        if rc == 0:
            return True
        time.sleep(10)
    return False


def get_kv_info():
    rc, out, _ = run_cmd("docker compose --profile qwen logs qwen --tail 100", timeout=15)
    info = {}
    for line in out.split('\n'):
        if 'KV cache size:' in line:
            try: info['kv_tokens'] = int(line.split('tokens')[0].split()[-1].replace(',', ''))
            except: pass
        if 'Maximum concurrency' in line:
            try:
                # "Maximum concurrency for 16,384 tokens per request: 12.43x"
                match = re.search(r'(\d+[,]*\d*)\s+tokens per request:\s+([\d.]+)x', line)
                if match:
                    info['max_model_len'] = int(match.group(1).replace(',', ''))
                    info['concurrency'] = float(match.group(2))
            except: pass
        if 'Available KV cache memory:' in line:
            try: info['kv_mem'] = line.split('Available KV cache memory:')[1].strip()
            except: pass
        if 'OOM' in line or 'out of memory' in line:
            info['oom'] = True
        if 'GPU Hang' in line or 'HW Exception' in line:
            info['crash'] = True
        if 'unsupported' in line.lower() and 'kv_cache' in line.lower():
            info['unsupported'] = True
    return info


def quick_bench():
    """Quick 3-turn benchmark."""
    import urllib.request, json
    turns = [
        "Explain the CAP theorem briefly.",
        "Now design a distributed key-value store for it.",
        "Implement a gossip protocol for this in Python.",
    ]
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    all_tps = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        data = json.dumps({
            "model": "model",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        elapsed = time.perf_counter() - start
        tokens = result.get('usage', {}).get('completion_tokens', 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        all_tps.append(tps)
        content = result['choices'][0]['message'].get('content', '') or ''
        messages.append({"role": "assistant", "content": content[:300]})
    
    return {
        'avg_tps': sum(all_tps) / len(all_tps),
        'min_tps': min(all_tps),
        'max_tps': max(all_tps),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-dtype", default="int8_per_token_head")
    parser.add_argument("--min-concurrency", type=float, default=4.0)
    parser.add_argument("--start-len", type=int, default=16384)
    parser.add_argument("--max-len", type=int, default=131072)
    args = parser.parse_args()

    # Test lengths: 16K, 32K, 48K, 64K, 96K, 128K
    lengths = []
    current = args.start_len
    while current <= args.max_len:
        lengths.append(current)
        current = int(current * 1.5)
        # Clean up to nice round numbers
        if current > 32768:
            current = (current // 8192) * 8192
    lengths = sorted(set(lengths))
    
    print(f"Testing kv-dtype={args.kv_dtype}, lengths={lengths}")
    print(f"Target: min concurrency >= {args.min_concurrency}x")
    print()
    print(f"{'Length':>10} {'Status':<10} {'Concur':>8} {'KV Mem':>10} {'KV Tokens':>12} {'Avg TPS':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

    best_len = args.start_len
    results = []

    for length in lengths:
        patch_config(args.kv_dtype, length)
        
        if not restart():
            print(f"{length:>10} {'START_FAIL':<10}")
            break
        
        if not wait_healthy():
            info = get_kv_info()
            if info.get('oom') or info.get('unsupported'):
                print(f"{length:>10} {'OOM/UNSUP':<10}")
            else:
                print(f"{length:>10} {'UNHEALTHY':<10}")
            run_cmd("docker compose --profile qwen down", timeout=30)
            break
        
        time.sleep(3)
        info = get_kv_info()
        concurrency = info.get('concurrency', 0)
        kv_mem = info.get('kv_mem', 'N/A')
        kv_tokens = info.get('kv_tokens', 'N/A')
        
        # Quick benchmark
        bench = quick_bench()
        avg_tps = bench['avg_tps']
        
        status = "OK" if concurrency >= args.min_concurrency else "LOW_CONC"
        print(f"{length:>10} {status:<10} {concurrency:>8.2f}x {kv_mem:>10} {kv_tokens:>12} {avg_tps:>10.1f}")
        
        results.append({
            'length': length,
            'concurrency': concurrency,
            'kv_mem': kv_mem,
            'kv_tokens': kv_tokens,
            'avg_tps': avg_tps,
        })
        
        if concurrency >= args.min_concurrency:
            best_len = length
        
        # If concurrency drops below 2x, stop - no point going higher
        if concurrency < 2.0:
            print("  Concurrency below 2x - stopping")
            break
        
        run_cmd("docker compose --profile qwen down", timeout=30)
    
    print(f"\nBest length for >= {args.min_concurrency}x concurrency: {best_len}")
    print(f"\nAll results:")
    for r in results:
        print(f"  {r['length']:>8} tokens: {r['concurrency']:.2f}x concur, {r['avg_tps']:.1f} tok/s avg")
