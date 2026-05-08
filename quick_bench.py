#!/usr/bin/env python3
"""Quick single-config benchmark. Tests one KV dtype, restarts container, runs benchmark.

Usage:
  python3 quick_bench.py fp8
  python3 quick_bench.py fp8_per_token_head
"""
import subprocess
import time
import sys
import re
import os

COMPOSE_DIR = "/home/pat/code/vllm-rocm"
COMPOSE_FILE = os.path.join(COMPOSE_DIR, "docker-compose.yml")


def run_cmd(cmd, timeout=300):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=COMPOSE_DIR)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def patch_kv_dtype(dtype):
    with open(COMPOSE_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'--kv-cache-dtype\s+\S+', f'--kv-cache-dtype {dtype}', content)
    with open(COMPOSE_FILE, 'w') as f:
        f.write(content)


def patch_max_model_len(length):
    with open(COMPOSE_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'--max-model-len\s+\S+', f'--max-model-len {length}', content)
    with open(COMPOSE_FILE, 'w') as f:
        f.write(content)


def restart():
    run_cmd("docker compose --profile qwen down", timeout=30)
    time.sleep(2)
    rc, _, err = run_cmd("docker compose --profile qwen up -d --force-recreate", timeout=60)
    if rc != 0:
        print(f"START FAILED: {err}")
        return False
    return True


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
            try: info['concurrency'] = line.split('concurrency')[1].strip().split()[0]
            except: pass
        if 'Available KV cache memory:' in line:
            try: info['kv_mem'] = line.split('Available KV cache memory:')[1].strip()
            except: pass
        if 'Cannot use' in line and 'paged attention' in line:
            info['paged_attn_warning'] = True
        if 'GPU Hang' in line or 'HW Exception' in line:
            info['crash'] = True
    return info


def bench(dtype):
    """Run the agent sim benchmark and return parsed results."""
    rc, out, err = run_cmd(
        f"python3 {COMPOSE_DIR}/config_bench.py --label 'kv={dtype}' --turns 7 --max-tokens 200",
        timeout=180
    )
    if rc != 0:
        return {'error': err[:500]}
    
    result = {'dtype': dtype}
    for line in out.split('\n'):
        if 'Overall TPS:' in line:
            try: result['overall_tps'] = float(line.split('Overall TPS:')[1].split('tok')[0].strip())
            except: pass
        if 'Single-stream' in line and 'tok/s' in line:
            try: result['single_tps'] = float(line.split('=')[1].split('tok/s')[0].strip())
            except: pass
        if 'min=' in line and 'max=' in line and 'avg=' in line:
            try:
                m = re.search(r'min=([\d.]+)\s+avg=([\d.]+)\s+max=([\d.]+)', line)
                if m:
                    result['min_tps'] = float(m.group(1))
                    result['avg_tps'] = float(m.group(2))
                    result['max_tps'] = float(m.group(3))
            except: pass
    return result


if __name__ == "__main__":
    dtype = sys.argv[1] if len(sys.argv) > 1 else "fp8"
    
    print(f"=== Testing kv-cache-dtype={dtype} ===")
    patch_kv_dtype(dtype)
    
    print("Restarting...")
    if not restart():
        print("FAILED to start container")
        sys.exit(1)
    
    print("Waiting for healthy...")
    if not wait_healthy():
        print("FAILED health check - checking logs")
        info = get_kv_info()
        print(f"KV info: {info}")
        _, logs, _ = run_cmd("docker compose --profile qwen logs qwen --tail 30", timeout=15)
        print(logs)
        sys.exit(1)
    
    time.sleep(3)
    info = get_kv_info()
    print(f"KV info: {info}")
    
    if info.get('crash'):
        print("CRASH DETECTED - skipping benchmark")
        print(json.dumps({'dtype': dtype, 'status': 'CRASH', 'kv_info': info}))
        sys.exit(1)
    
    result = bench(dtype)
    result['kv_info'] = info
    result['status'] = 'OK' if 'error' not in result else 'BENCH_ERROR'
    
    print(f"\n=== RESULT: {dtype} ===")
    print(f"  Overall TPS: {result.get('overall_tps', 'N/A')}")
    print(f"  Single TPS:  {result.get('single_tps', 'N/A')}")
    print(f"  Avg TPS:     {result.get('avg_tps', 'N/A')}")
    print(f"  Min TPS:     {result.get('min_tps', 'N/A')}")
    print(f"  Max TPS:     {result.get('max_tps', 'N/A')}")
    print(f"  Concurrency: {info.get('concurrency', 'N/A')}")
    print(f"  KV tokens:   {info.get('kv_tokens', 'N/A')}")
    print(f"  KV mem:      {info.get('kv_mem', 'N/A')}")
