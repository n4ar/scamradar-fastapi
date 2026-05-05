import aiosqlite
import hashlib
import os

DB_PATH = "data/votes.db"

async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS community_votes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                msg_hash    TEXT NOT NULL,
                vote        TEXT NOT NULL CHECK(vote IN ('scam', 'safe')),
                user_hash   TEXT NOT NULL,
                bigru_score REAL,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        # UNIQUE ป้องกัน 1 user vote ซ้ำ message เดิม
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_user
            ON community_votes(msg_hash, user_hash)
        """)
        await db.commit()

async def save_vote(msg_hash: str, vote: str, user_id: str, bigru_score: float = None):
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO community_votes(msg_hash, vote, user_hash, bigru_score) VALUES(?,?,?,?)",
                (msg_hash, vote, user_hash, bigru_score)
            )
            await db.commit()
            print(f"Vote saved: {vote} for {msg_hash} by {user_hash[:8]}")
        except aiosqlite.IntegrityError:
            print(f"Vote skipped: User {user_hash[:8]} already voted on {msg_hash}")
            pass  # user vote ซ้ำ — ไม่ทำอะไร
