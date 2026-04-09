CREATE TABLE chain_state (
    job_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    updated_at REAL NOT NULL
);
