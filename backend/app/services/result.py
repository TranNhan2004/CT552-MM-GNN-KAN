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
                processed_texts TEXT,
                
                cnn_label_idx INTEGER,
                cnn_label_name TEXT,
                cnn_prob REAL,
                
                img_txt_label_idx INTEGER,
                img_txt_label_name TEXT,
                img_txt_prob REAL,
                img_txt_weights TEXT,
                
                full_label_idx INTEGER,
                full_label_name TEXT,
                full_prob REAL,
                full_weights TEXT
            )
            """)
            conn.commit()

    def insert(self, value: ResultModel) -> ResultModel:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO result (
                image_urls, audio_urls, text, processed_texts,
                cnn_label_idx, cnn_label_name, cnn_prob,
                img_txt_label_idx, img_txt_label_name, img_txt_prob, img_txt_weights,
                full_label_idx, full_label_name, full_prob, full_weights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(value.image_urls),
                json.dumps(value.audio_urls),
                value.text,
                json.dumps(value.processed_texts),
                
                value.img_label_idx,
                value.img_label_name,
                value.img_prob,
                
                value.img_txt_label_idx,
                value.img_txt_label_name,
                value.img_txt_prob,
                json.dumps(value.img_txt_weights),
                
                value.full_label_idx,
                value.full_label_name,
                value.full_prob,
                json.dumps(value.full_weights)
            ))
            conn.commit()
            value.id = cursor.lastrowid
        return value

    def update(self, value: ResultModel) -> Optional[ResultModel]:
        if value.id is None:
            raise ValueError("ResultModel.id must be set to update record.")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE result
            SET image_urls = ?,
                audio_urls = ?,
                text = ?,
                processed_texts = ?,
                
                cnn_label_idx = ?,
                cnn_label_name = ?,
                cnn_prob = ?,
                
                img_txt_label_idx = ?,
                img_txt_label_name = ?,
                img_txt_prob = ?,
                img_txt_weights = ?,
                
                full_label_idx = ?,
                full_label_name = ?,
                full_prob = ?,
                full_weights = ?
            WHERE id = ?
            """, (
                json.dumps(value.image_urls),
                json.dumps(value.audio_urls),
                value.text,
                json.dumps(value.processed_texts),
                
                value.img_label_idx,
                value.img_label_name,
                value.img_prob,
                
                value.img_txt_label_idx,
                value.img_txt_label_name,
                value.img_txt_prob,
                json.dumps(value.img_txt_weights),
                
                value.full_label_idx,
                value.full_label_name,
                value.full_prob,
                json.dumps(value.full_weights),
                
                value.id
            ))
            conn.commit()
            
            if cursor.rowcount == 0:
                return None
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

                img_label_idx=row[5],
                img_label_name=row[6],
                img_prob=row[7],
                
                img_txt_label_idx=row[8],
                img_txt_label_name=row[9],
                img_txt_prob=row[10],
                img_txt_weights=json.loads(row[11]) if row[11] else [],
                
                full_label_idx=row[12],
                full_label_name=row[13],
                full_prob=row[14],
                full_weights=json.loads(row[15]) if row[15] else []
            )

    def find_last_id(self) -> int | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='result'")
            row = cursor.fetchone()
            return row[0] if row else None