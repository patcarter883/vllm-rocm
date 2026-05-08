#!/usr/bin/env python3
"""Reproduce the long-context decode slowdown that OpenCode triggers."""
import aiohttp
import asyncio
import time
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "model"

# Generate a very long system prompt to simulate OpenCode's context
# ~18000 tokens = ~72000 chars roughly
SYSTEM_PROMPT_LARGE = """You are an expert AI coding assistant. You have deep knowledge of software engineering, system design, and multiple programming languages. You follow these principles:

1. CODE QUALITY: Write clean, well-documented code with proper error handling. Use type hints in Python, proper const correctness in C++, and idiomatic patterns in each language. Always consider edge cases and write defensive code.

2. ARCHITECTURE: When designing systems, consider scalability, reliability, and maintainability. Document trade-offs explicitly. Use established patterns (CQRS, Event Sourcing, Saga, etc.) where appropriate.

3. SECURITY: Always consider security implications. Validate inputs, use parameterized queries, implement proper authentication and authorization, and follow the principle of least privilege.

4. PERFORMANCE: Profile before optimizing. Consider algorithmic complexity, memory usage, and I/O patterns. Use appropriate data structures and avoid premature optimization.

5. TESTING: Write comprehensive tests including unit tests, integration tests, and edge cases. Use test-driven development when appropriate.

Here are the project files you have access to:

--- src/main.rs ---
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{Router, routing::get, extract::State};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct Task {
    id: u64,
    payload: String,
    priority: i32,
    status: TaskStatus,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
enum TaskStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

struct AppState {
    tasks: Arc<RwLock<Vec<Task>>>,
    next_id: Arc<tokio::sync::Mutex<u64>>,
}

#[tokio::main]
async fn main() {
    let state = AppState {
        tasks: Arc::new(RwLock::new(Vec::new())),
        next_id: Arc::new(tokio::sync::Mutex::new(1)),
    };
    
    let app = Router::new()
        .route("/tasks", get(list_tasks).post(create_task))
        .route("/tasks/:id", get(get_task).put(update_task).delete(delete_task))
        .route("/tasks/:id/claim", post(claim_task))
        .route("/health", get(health_check))
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

--- src/db.rs ---
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use crate::models::Task;

pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(20)
        .connect(database_url)
        .await
}

pub async fn claim_tasks(pool: &PgPool, worker_id: &str, batch_size: i32) -> Result<Vec<Task>, sqlx::Error> {
    let mut tx = pool.begin().await?;
    let tasks = sqlx::query_as!(
        Task,
        "UPDATE tasks SET status = 'processing', worker_id = $1, claimed_at = NOW() \
         WHERE id IN (SELECT id FROM tasks WHERE status = 'pending' \
         ORDER BY priority DESC, created_at ASC LIMIT $2 FOR UPDATE SKIP LOCKED) \
         RETURNING *",
        worker_id, batch_size
    ).fetch_all(&mut *tx).await?;
    tx.commit().await?;
    Ok(tasks)
}

--- src/models.rs ---
use serde::{Deserialize, Serialize};
use chrono::NaiveDateTime;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Task {
    pub id: i64,
    pub payload: serde_json::Value,
    pub priority: i32,
    pub status: String,
    pub worker_id: Option<String>,
    pub created_at: NaiveDateTime,
    pub claimed_at: Option<NaiveDateTime>,
    pub completed_at: Option<NaiveDateTime>,
    pub visibility_timeout_secs: Option<i32>,
}

--- src/worker.rs ---
use tokio::time::{sleep, Duration};
use crate::db::claim_tasks;
use crate::processor::process_task;

pub async fn worker_loop(pool: PgPool, worker_id: String) {
    loop {
        match claim_tasks(&pool, &worker_id, 10).await {
            Ok(tasks) if tasks.is_empty() => {
                sleep(Duration::from_millis(500)).await;
            }
            Ok(tasks) => {
                for task in tasks {
                    if let Err(e) = process_task(&pool, &task).await {
                        eprintln!("Task {} failed: {}", task.id, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to claim tasks: {}", e);
                sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

--- Cargo.toml ---
[package]
name = "task-scheduler"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres", "chrono"] }
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = "0.3"
thiserror = "1"

--- migrations/001_init.sql ---
CREATE TABLE tasks (
    id BIGSERIAL PRIMARY KEY,
    payload JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    worker_id VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMP,
    completed_at TIMESTAMP,
    visibility_timeout_secs INTEGER DEFAULT 300
);

CREATE INDEX idx_tasks_status_priority ON tasks (status, priority DESC, created_at) WHERE status = 'pending';
CREATE INDEX idx_tasks_worker ON tasks (worker_id) WHERE status = 'processing';

--- Dockerfile ---
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/task-scheduler /usr/local/bin/
CMD ["task-scheduler"]

--- docker-compose.yml ---
version: '3.8'
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgres://scheduler:scheduler@db:5432/scheduler
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: scheduler
      POSTGRES_PASSWORD: scheduler
      POSTGRES_DB: scheduler
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d

volumes:
  pgdata:
"""

# Repeat the system prompt to reach ~18K tokens
SYSTEM_PROMPT_HUGE = SYSTEM_PROMPT_LARGE + "\n\n" + SYSTEM_PROMPT_LARGE + "\n\n" + SYSTEM_PROMPT_LARGE

async def single_stream_bench(session, system_prompt, user_msg, max_tokens, label):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
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
    
    print(f"  {label:25s}: TTFT={ttft:6.0f}ms  TPS={tps:5.1f}  tokens={token_count:4d}  gen_time={gen_time:.2f}s")
    return {"label": label, "ttft_ms": ttft, "tps": tps, "tokens": token_count}

async def main():
    # Estimate token counts
    sp1_len = len(SYSTEM_PROMPT_LARGE)  
    sp3_len = len(SYSTEM_PROMPT_HUGE)
    print(f"System prompt (1x): ~{sp1_len} chars (~{sp1_len//4} tokens)")
    print(f"System prompt (3x): ~{sp3_len} chars (~{sp3_len//4} tokens)")
    print()
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up...")
        for _ in range(3):
            payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20, "stream": True}
            async with session.post(API_URL, json=payload) as resp:
                async for line in resp.content:
                    pass
        print()
        
        # Test different context sizes
        print("Context size vs decode speed:")
        print("-" * 80)
        
        # Short context (no system prompt)
        await single_stream_bench(session, "", "What is 2+2?", 100, "no_system_100tok")
        
        # 1x system prompt
        await single_stream_bench(session, SYSTEM_PROMPT_LARGE, "Explain briefly what this project does.", 100, "1x_system_100tok")
        await single_stream_bench(session, SYSTEM_PROMPT_LARGE, "Explain briefly what this project does.", 500, "1x_system_500tok")
        
        # 3x system prompt (~18K tokens) - reproducing OpenCode scenario
        await single_stream_bench(session, SYSTEM_PROMPT_HUGE, "What pattern does the worker loop use?", 100, "3x_system_100tok")
        await single_stream_bench(session, SYSTEM_PROMPT_HUGE, "What pattern does the worker loop use?", 500, "3x_system_500tok")

if __name__ == "__main__":
    asyncio.run(main())
