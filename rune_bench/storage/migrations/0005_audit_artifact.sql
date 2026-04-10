-- migrate:sqlite
CREATE TABLE audit_artifact (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    content BLOB NOT NULL,
    created_at REAL NOT NULL
);

-- migrate:postgres
CREATE TABLE audit_artifact (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    content BYTEA NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

-- migrate:all
CREATE INDEX idx_audit_artifact_job_id ON audit_artifact(job_id);
