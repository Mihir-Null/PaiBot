-- Static table DDL for nanobot memory index.
-- chunks_vec is created separately in code (dimension is configurable).

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id    INTEGER PRIMARY KEY,
    path  TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    hash  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER REFERENCES files(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    created_at REAL NOT NULL,
    embedding  BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    tokenize='porter unicode61'
);
