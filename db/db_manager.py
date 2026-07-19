"""
db/db_manager.py
================
SQLite Database Manager for storing enrolled user identity embeddings.
Uses standard library sqlite3 for portability and numpy for serialisation.
"""

import sqlite3
import os
from typing import List, Tuple, Dict, Optional
import numpy as np


class DatabaseManager:
    """
    Manages connection and CRUD operations for the face embedding database.
    """

    def __init__(self, db_path: str = "embeddings.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Embeddings table (allows multiple templates/samples per user)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # 3. Portal credentials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portal_credentials (
                email TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        conn.close()

    def register_user(self, username: str) -> int:
        """
        Creates a new user profile. Returns user_id.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username) VALUES (?);", (username,))
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            # User already exists, retrieve their ID
            cursor.execute("SELECT id FROM users WHERE username = ?;", (username,))
            row = cursor.fetchone()
            return row[0]
        finally:
            conn.close()

    def add_user_embedding(self, user_id: int, embedding: np.ndarray) -> None:
        """
        Stores a face embedding template.
        """
        # Convert float32 numpy array to raw bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?);",
            (user_id, embedding_bytes)
        )
        conn.commit()
        conn.close()

    def get_user_id(self, username: str) -> Optional[int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?;", (username,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_user_embeddings(self, user_id: int) -> List[np.ndarray]:
        """
        Retrieves all registered embedding templates for a specific user ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE user_id = ?;", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        embeddings = []
        for row in rows:
            # Convert raw bytes back to numpy float32 array
            arr = np.frombuffer(row[0], dtype=np.float32)
            embeddings.append(arr)
        return embeddings

    def get_all_users_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """
        Retrieves the entire enrolled library for 1:N recognition matching.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.username, e.embedding 
            FROM users u 
            JOIN embeddings e ON u.id = e.user_id;
        """)
        rows = cursor.fetchall()
        conn.close()

        library: Dict[str, List[np.ndarray]] = {}
        for username, emb_bytes in rows:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            if username not in library:
                library[username] = []
            library[username].append(arr)
        return library

    def delete_user(self, username: str) -> bool:
        """Deletes user and cascades to delete all their embeddings."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?;", (username,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count > 0

    def clear_database(self) -> None:
        """Wipe all registered users and portal credentials."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users;")
        try:
            cursor.execute("DELETE FROM portal_credentials;")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()

    def register_portal_user(self, email: str, username: str, password_hash: str) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO portal_credentials (email, username, password_hash) VALUES (?, ?, ?);",
            (email, username, password_hash)
        )
        conn.commit()
        conn.close()

    def get_portal_user(self, email: str) -> Optional[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT email, username, password_hash FROM portal_credentials WHERE email = ?;", (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"email": row[0], "username": row[1], "password_hash": row[2]}
        return None
