import aiosqlite
import hashlib
import os
import re

DB_PATH = "data/votes.db"

def redact_pii(text: str) -> str:
    if not text:
        return text
    # Redact phone numbers (e.g., 081-123-4567, 0811234567, +66811234567)
    text = re.sub(r'(\+66|0)\s?[689]\s?\d\s?-?\s?\d{3}\s?-?\s?\d{4}', '[PHONE_REDACTED]', text)
    # Redact potential bank accounts/national IDs (10-13 consecutive digits)
    text = re.sub(r'\b\d{10,13}\b', '[ID/ACC_REDACTED]', text)
    return text

async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        # Table 1: Messages (Logs the analysis once per unique message)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                msg_hash      TEXT NOT NULL UNIQUE,
                original_text TEXT,
                bigru_score   REAL,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
        # Table 2: Votes (Logs community votes linked to the message)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                msg_hash    TEXT NOT NULL,
                vote        TEXT NOT NULL CHECK(vote IN ('scam', 'safe')),
                user_hash   TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (msg_hash) REFERENCES messages(msg_hash)
            )
        """)
        # Prevent 1 user from voting on the same message multiple times
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_user_vote
            ON votes(msg_hash, user_hash)
        """)
        await db.commit()

async def log_analysis(msg_hash: str, text: str, score: float):
    safe_text = redact_pii(text)
    async with aiosqlite.connect(DB_PATH) as db:
        # INSERT OR IGNORE means if the msg_hash already exists, it skips quietly
        await db.execute(
            "INSERT OR IGNORE INTO messages(msg_hash, original_text, bigru_score) VALUES(?,?,?)",
            (msg_hash, safe_text, score)
        )
        await db.commit()

async def save_vote(msg_hash: str, vote: str, user_id: str, bigru_score: float = None) -> bool:
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO votes(msg_hash, vote, user_hash) VALUES(?,?,?)",
                (msg_hash, vote, user_hash)
            )
            await db.commit()
            print(f"Vote saved: {vote} for {msg_hash} by {user_hash[:8]}")
            return True
        except aiosqlite.IntegrityError:
            print(f"Vote skipped: User {user_hash[:8]} already voted on {msg_hash}")
            return False
