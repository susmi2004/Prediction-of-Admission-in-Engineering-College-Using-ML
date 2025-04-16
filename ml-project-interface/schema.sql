CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    location TEXT NOT NULL,
    tenth_percentage REAL NOT NULL,
    inter_percentage REAL NOT NULL,
    category TEXT NOT NULL,
    student_mobile TEXT NOT NULL,
    parent_mobile TEXT NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
