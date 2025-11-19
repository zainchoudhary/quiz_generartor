import sqlite3
from datetime import datetime
from typing import List, Dict

DB_PATH = "quiz_app.db"

# ------------------- Initialize Database -------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Quizzes table
    c.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            quiz_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

# ------------------- User Auth -------------------
def register_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username: str, password: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, password))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1]}
    return None

# ------------------- Quiz Storage -------------------
import json

def save_quiz(user_id: int, quiz_data: List[Dict]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quiz_json = json.dumps(quiz_data)
    c.execute("INSERT INTO quizzes (user_id, quiz_data, created_at) VALUES (?, ?, ?)",
              (user_id, quiz_json, created_at))
    conn.commit()
    conn.close()

def get_quiz_history(user_id: int) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT quiz_data, created_at FROM quizzes WHERE user_id=? ORDER BY id ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    history = []
    for quiz_json, created_at in rows:
        quiz_data = json.loads(quiz_json)
        history.append({"quiz_data": quiz_data, "created_at": created_at})
    return history
