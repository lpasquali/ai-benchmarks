-- migrate:sqlite
CREATE TABLE workflow_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    event TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL,
    error_type TEXT,
    labels_json TEXT,
    recorded_at REAL NOT NULL
);

-- migrate:postgres
CREATE TABLE workflow_events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT,
    event TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION,
    error_type TEXT,
    labels_json TEXT,
    recorded_at DOUBLE PRECISION NOT NULL
);

-- migrate:all
CREATE INDEX idx_workflow_events_job_id ON workflow_events(job_id);

CREATE INDEX idx_workflow_events_event ON workflow_events(event);
