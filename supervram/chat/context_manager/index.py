from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Iterable


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    content_hash: str
    content: str
    language: str


@dataclass(frozen=True)
class SearchResult:
    chunk: CodeChunk
    score: float
    source: str = "fts5"


class RepositoryIndex:
    def __init__(self, db_path: str | Path, repo_root: str | Path, excludes: Iterable[str], chunk_lines: int = 80, overlap_lines: int = 12):
        self.db_path = Path(db_path)
        self.repo_root = Path(repo_root).resolve()
        self.excludes = tuple(excludes)
        self.chunk_lines = chunk_lines
        self.overlap_lines = min(overlap_lines, max(0, chunk_lines - 1))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        # This file is shared across every session on the same workspace (index_db has no
        # session component), but each session gets its own ContextManager/RepositoryIndex/
        # sqlite3 connection with its own Python-level lock -- so ContextManager's `_lock` only
        # serializes calls *within* one session, not a background indexing session against a
        # concurrent chat session's retrieval reads on the same file. Without a busy_timeout,
        # SQLite's default journal mode fails a conflicting access immediately ("database is
        # locked") rather than waiting, and a badly-timed interleaving (observed once here)
        # produced "database disk image is malformed" instead.
        #
        # WAL mode was tried first (SQLite's usual answer for one writer + concurrent readers)
        # and made things worse, not better: it killed the whole server process with SIGBUS
        # within seconds under the exact same concurrent-access test that this pragma alone
        # handles cleanly. WAL relies on a shared-memory (-shm) coordination file between
        # connections, and something about this sandboxed environment doesn't tolerate that.
        # busy_timeout alone doesn't get WAL's true multi-reader concurrency, but it turns
        # "SQLITE_BUSY fails immediately" into "wait and retry," which is enough for this app's
        # actual access pattern (one background indexer, occasional concurrent reads) without
        # the crash risk.
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                language TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                path UNINDEXED, start_line UNINDEXED, end_line UNINDEXED,
                symbol, content_hash UNINDEXED, language UNINDEXED, content,
                tokenize='unicode61 tokenchars ''_'''
            );
        """)
        self.connection.commit()

    def _excluded(self, relative: str) -> bool:
        return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch("/" + relative, pattern) for pattern in self.excludes)

    def index(self) -> dict:
        indexed = 0
        unchanged = 0
        removed = 0
        seen: set[str] = set()
        for path in self.repo_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve().relative_to(self.repo_root)
            except ValueError:
                continue
            relative = path.relative_to(self.repo_root).as_posix()
            if self._excluded(relative) or path.stat().st_size > 2_000_000:
                continue
            language = self._language(path)
            if language == "binary":
                continue
            seen.add(relative)
            stat = path.stat()
            old = self.connection.execute("SELECT content_hash,mtime_ns,size FROM files WHERE path=?", (relative,)).fetchone()
            if old and old["mtime_ns"] == stat.st_mtime_ns and old["size"] == stat.st_size:
                unchanged += 1
                continue
            raw = path.read_bytes()
            text = raw.decode(errors="replace")
            digest = hashlib.sha256(raw).hexdigest()
            if old and old["content_hash"] == digest:
                self.connection.execute("UPDATE files SET mtime_ns=?,size=? WHERE path=?", (stat.st_mtime_ns, stat.st_size, relative))
                unchanged += 1
                continue
            self.connection.execute("DELETE FROM chunks WHERE path=?", (relative,))
            self.connection.execute(
                "INSERT OR REPLACE INTO files(path,content_hash,mtime_ns,size,language) VALUES(?,?,?,?,?)",
                (relative, digest, stat.st_mtime_ns, stat.st_size, language),
            )
            for chunk in self._chunks(relative, text, digest, language):
                self.connection.execute(
                    "INSERT INTO chunks(path,start_line,end_line,symbol,content_hash,language,content) VALUES(?,?,?,?,?,?,?)",
                    (chunk.path, chunk.start_line, chunk.end_line, chunk.symbol or "", chunk.content_hash, chunk.language, chunk.content),
                )
            indexed += 1
        rows = self.connection.execute("SELECT path FROM files").fetchall()
        for row in rows:
            if row["path"] not in seen:
                self.connection.execute("DELETE FROM files WHERE path=?", (row["path"],))
                self.connection.execute("DELETE FROM chunks WHERE path=?", (row["path"],))
                removed += 1
        self.connection.commit()
        return {"indexed": indexed, "unchanged": unchanged, "removed": removed, "files": len(seen)}

    def search(self, query: str, limit: int = 12) -> list[SearchResult]:
        terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms[:20]))
        rows = self.connection.execute(
            "SELECT path,start_line,end_line,symbol,content_hash,language,content,bm25(chunks,1.0,0.0,0.0,4.0,0.0,0.0,1.0) AS rank FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (expression, limit),
        ).fetchall()
        results = []
        for row in rows:
            chunk = CodeChunk(row["path"], int(row["start_line"]), int(row["end_line"]), row["symbol"] or None, row["content_hash"], row["content"], row["language"])
            results.append(SearchResult(chunk, -float(row["rank"])))
        return results

    def validate(self, chunk: CodeChunk) -> bool:
        path = self.repo_root / chunk.path
        if not path.exists():
            return False
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        return current == chunk.content_hash

    def refresh_path(self, relative: str) -> None:
        path = self.repo_root / relative
        if not path.exists() or path.is_symlink():
            self.connection.execute("DELETE FROM files WHERE path=?", (relative,))
            self.connection.execute("DELETE FROM chunks WHERE path=?", (relative,))
            self.connection.commit()
            return
        stat = path.stat()
        raw = path.read_bytes()
        text = raw.decode(errors="replace")
        digest = hashlib.sha256(raw).hexdigest()
        language = self._language(path)
        self.connection.execute("DELETE FROM chunks WHERE path=?", (relative,))
        self.connection.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?)", (relative, digest, stat.st_mtime_ns, stat.st_size, language))
        for chunk in self._chunks(relative, text, digest, language):
            self.connection.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", (chunk.path, chunk.start_line, chunk.end_line, chunk.symbol or "", chunk.content_hash, chunk.language, chunk.content))
        self.connection.commit()

    def _chunks(self, relative: str, text: str, digest: str, language: str) -> list[CodeChunk]:
        lines = text.splitlines()
        symbols = self._symbols(text, language)
        result = []
        step = max(1, self.chunk_lines - self.overlap_lines)
        for start in range(0, len(lines) or 1, step):
            end = min(len(lines), start + self.chunk_lines)
            symbol = next((name for line, name in reversed(symbols) if line <= start + 1), None)
            content = "\n".join(lines[start:end])
            if content.strip():
                result.append(CodeChunk(relative, start + 1, end, symbol, digest, content, language))
            if end == len(lines):
                break
        return result

    @staticmethod
    def _symbols(text: str, language: str) -> list[tuple[int, str]]:
        if language == "python":
            try:
                tree = ast.parse(text)
                return sorted((node.lineno, node.name) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            except SyntaxError:
                return []
        patterns = {
            "javascript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)", re.M),
            "cpp": re.compile(r"^\s*(?:class|struct)\s+(\w+)|^\s*[\w:<>,*&\s]+\s+(\w+)\s*\([^;]*\)\s*\{", re.M),
        }
        pattern = patterns.get(language)
        if not pattern:
            return []
        symbols = []
        for match in pattern.finditer(text):
            name = next((group for group in match.groups() if group), None)
            if name:
                symbols.append((text.count("\n", 0, match.start()) + 1, name))
        return symbols

    @staticmethod
    def _language(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".py": return "python"
        if suffix in {".js", ".ts", ".tsx", ".jsx", ".html", ".svelte"}: return "javascript"
        if suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}: return "cpp"
        if suffix in {".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".sh", ".css"}: return "text"
        try:
            path.read_text(errors="strict")
            return "text"
        except (UnicodeDecodeError, OSError):
            return "binary"

    def close(self) -> None:
        self.connection.close()
