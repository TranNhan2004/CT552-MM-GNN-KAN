from pathlib import Path
import sqlite3
import json

from typing import Optional
from ..models.result import ResultModel

class ResultService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._create_table()

    def _create_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_urls TEXT,
                audio_urls TEXT,
                text TEXT,
                processed_text TEXT,
                label_idx INTEGER,
                label_name TEXT,
                prob REAL,
                weights TEXT
            )
            """)
            conn.commit()

    def insert(self, value: ResultModel) -> ResultModel:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO result (
                image_urls, audio_urls, text, processed_text, label_idx, label_name, prob, weights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(value.image_urls),
                json.dumps(value.audio_urls),
                value.text,
                json.dumps(value.processed_text),
                value.label_idx,
                value.label_name,
                value.prob,
                json.dumps(value.weights) if value.weights else None
            ))
            conn.commit()
            value.id = cursor.lastrowid
            return value

    def find(self, id: int) -> Optional[ResultModel]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM result WHERE id = ?", (id,))
            row = cursor.fetchone()
            if not row:
                return None

            return ResultModel(
                id=row[0],
                image_urls=json.loads(row[1]),
                audio_urls=json.loads(row[2]),
                text=row[3],
                processed_texts=json.loads(row[4]),
                label_idx=row[5],
                label_name=row[6],
                prob=row[7],
                weights=json.loads(row[8]) if row[8] else None
            )

    def find_last_id(self) -> int | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='result'")
            row = cursor.fetchone()
            return row[0] if row else None