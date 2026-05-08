#!/usr/bin/env python3
"""Cycle through KV cache quantization configs, restarting vLLM each time.

Tests: fp8, fp8_e4m3, fp8_e5m2, fp8_per_token_head, int8_per_token_head, float16
For each: start container, wait for health, run benchmark, collect metrics, stop container.
"""
import subprocess
import time
import json
import sys
import os

COMPOSE_DIR = "/home/pat/code/vllm-rocm"
COMPOSE_FILE = os.path.join(COMPOSE_DIR, "docker-compose.yml")
BENCH_SCRIPT = os.path.join(COMPOSE_DIR, "config_bench.py")

# KV cache dtypes to test (ROCm-compatible ones only)
KV_DTYPES = [
    "fp8",
    "fp8_e4m3", 
    "fp8_e5m2",
    "fp8_per_token_head",
    "int8_per_token_head",
    "float16",
]

MAX_MODEL_LEN = 16384


def run_cmd(cmd, timeout=300):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=COMPOSE_DIR
    )
    return result.returncode, result.stdout, result.stderr


def patch_kv_dtype(dtype):
    """Update docker-compose.yml with the given kv-cache-dtype."""
    with open(COMPOSE_FILE, 'r') as f:
        content = f.read()
    
    # Replace --kv-cache-dtype line
    import re
    content = re.sub(
        r'--kv-cache-dtype\s+\S+',
        f'--kv-cache-dtype {dtype}',
        content
    )
    
    with open(COMPOSE_FILE, 'w') as f:
        f.write(content)
    print(f"  Patched kv-cache-dtype to: {dtype}")


def patch_max_model_len(length):
    """Update docker-compose.yml with the given max-model-len."""
    with open(COMPOSE_FILE, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(
        r'--max-model-len\s+\S+',
        f'--max-model-len {length}',
        content
    )
    
    with open(COMPOSE_FILE, 'w') as f:
        f.write(content)


def restart_container():
    """Recreate and start the container."""
    rc, out, err = run_cmd("docker compose --profile qwen down", timeout=30)
    if rc != 0:
        print(f"  WARNING: down failed: {err}")
    
    rc, out, err = run_cmd("docker compose --profile qwen up -d --force-recreate", timeout=60)
    if rc != 0:
        print(f"  ERROR: up failed: {err}")
        return False
    return True


def wait_for_healthy(max_wait=180):
    """Wait for the vLLM server to respond to /health."""
    start = time.time()
    while time.time() - start < max_wait:
        rc, out, err = run_cmd("curl -sf http://localhost:8000/health", timeout=10)
        if rc == 0:
            elapsed = time.time() - start
            print(f"  Server healthy after {elapsed:.0f}s")
            return True
        time.sleep(10)
    print(f"  ERROR: Server not healthy after {max_wait}s")
    return False


def get_kv_cache_info():
    """Get KV cache token count and concurrency from logs."""
    rc, out, err = run_cmd("docker compose --profile qwen logs qwen --tail 50", timeout=15)
    info = {}
    for line in out.split('\n'):
        if 'KV cache size:' in line:
            # "GPU KV cache size: 99,072 tokens"
            try:
                info['kv_tokens'] = int(line.split('tokens')[0].split()[-1].replace(',', ''))
            except:
                pass
        if 'Maximum concurrency' in line:
            # "Maximum concurrency for 16,384 tokens per request: 12.43x"
            try:
                parts = line.split('concurrency')[1]
                info['concurrency'] = parts.strip().split()[0]
            except:
                pass
        if 'Available KV cache memory:' in line:
            try:
                info['kv_mem_gib'] = line.split('Available KV cache memory:')[1].strip().split()[0]
            except:
                pass
    return info


def run_benchmark(label):
    """Run the benchmark script and return results."""
    rc, out, err = run_cmd(
        f"python3 {BENCH_SCRIPT} --label '{label}' --turns 7 --max-tokens 200",
        timeout=180
    )
    return out if rc == 0 else f"ERROR: {err}"


def main():
    results = []
    
    # Save original max_model_len
    patch_max_model_len(MAX_MODEL_LEN)
    
    for dtype in KV_DTYPES:
        print(f"\n{'='*60}")
        print(f"  Testing kv-cache-dtype: {dtype}")
        print(f"{'='*60}")
        
        # Patch config
        patch_kv_dtype(dtype)
        
        # Restart
        print("  Restarting container...")
        if not restart_container():
            results.append({'dtype': dtype, 'status': 'FAILED_START', 'error': 'Container failed to start'})
            continue
        
        # Wait for health
        if not wait_for_healthy():
            results.append({'dtype': dtype, 'status': 'FAILED_HEALTH', 'error': 'Server not healthy'})
            # Check logs for error
            rc, out, err = run_cmd("docker compose --profile qwen logs qwen --tail 20", timeout=15)
            print(f"  Last logs: {out[-500:]}")
            continue
        
        # Get KV cache info
        time.sleep(5)  # Let logs settle
        kv_info = get_kv_cache_info()
        print(f"  KV cache: {kv_info}")
        
        # Run benchmark
        print("  Running benchmark...")
        bench_output = run_benchmark(f"kv-dtype={dtype}")
        print(bench_output)
        
        # Parse benchmark results
        result = {
            'dtype': dtype,
            'status': 'OK',
            'kv_info': kv_info,
            'raw_output': bench_output[-1000:],  # Last 1000 chars
        }
        
        # Extract TPS from output
        for line in bench_output.split('\n'):
            if 'Overall TPS:' in line:
                try:
                    result['overall_tps'] = float(line.split('Overall TPS:')[1].split('tok')[0].strip())
                except:
                    pass
            if 'Per-turn TPS:' in line:
                try:
                    result['per_turn_tps'] = line.split('Per-turn TPS:')[1].strip()
                except:
                    pass
            if 'Single-stream' in line and 'tok/s' in line:
                try:
                    result['single_tps'] = float(line.split('=')[1].split('tok/s')[0].strip())
                except:
                    pass
        
        results.append(result)
        print(f"  Result: {result.get('overall_tps', 'N/A')} tok/s overall, "
              f"{result.get('single_tps', 'N/A')} tok/s single, "
              f"concurrency={kv_info.get('concurrency', 'N/A')}")
    
    # Clean up
    print(f"\n\n{'='*60}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'dtype':<25} {'status':<8} {'overall_tps':>12} {'single_tps':>12} {'concurrency':>12} {'kv_tokens':>12}")
    print(f"{'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for r in results:
        print(f"{r['dtype']:<25} {r['status']:<8} "
              f"{r.get('overall_tps', 'N/A'):>12} "
              f"{r.get('single_tps', 'N/A'):>12} "
              f"{r.get('kv_info', {}).get('concurrency', 'N/A'):>12} "
              f"{r.get('kv_info', {}).get('kv_tokens', 'N/A'):>12}")
    
    # Save results
    results_file = os.path.join(COMPOSE_DIR, "bench_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
