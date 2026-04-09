CREATE TABLE idempotency_keys (
    tenant_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    job_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, operation, idempotency_key)
);
