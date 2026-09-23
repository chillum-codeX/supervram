"""Coding-agent tools for SuperVRAM chat: read/list/search/propose-edit, all hard-sandboxed to a
chosen workspace folder. No shell/command execution tool exists here by design -- the blast radius
of an arbitrary-command tool is a different risk category than file read/write, and the user asked
for it to be left out. Writes are never applied directly: write_file only stages a diff; the actual
write happens through apply_write, called only after the person approves it in the UI.
"""
import difflib
import os
from pathlib import Path

MAX_READ_CHARS = 60_000       # keep a single file read within a reasonable prompt budget
MAX_SEARCH_MATCHES = 150
SKIP_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", "build", "build-3090", "build-supervram",
    ".venv", "venv", ".pytest_cache", "dist", "target", ".cache",
}
SKIP_SUFFIXES = {
    ".gguf", ".bin", ".so", ".o", ".a", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".woff", ".woff2", ".ttf", ".svg",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories at a path inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root. Use \".\" for the root."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's contents from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text query across files in the workspace (case-insensitive substring match). Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "description": "Subdirectory to search within, relative to the workspace root. Defaults to the whole workspace."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Propose creating a new file or overwriting an existing one with the given full "
                "content. This does NOT write to disk -- it stages the change as a diff for the "
                "person to review and approve or reject in the UI. Always pass the file's complete "
                "new content, not a partial patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "The full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
]


class SandboxError(Exception):
    pass


def safe_path(workspace_root: str, rel_path: str) -> Path:
    """Resolve rel_path against workspace_root and refuse anything that escapes it (via "..",
    an absolute path, or a symlink pointing outside). Raises SandboxError rather than returning
    a possibly-unsafe path."""
    base = Path(workspace_root).resolve()
    rel_path = rel_path or "."
    candidate = (base / rel_path) if not os.path.isabs(rel_path) else Path(rel_path)
    target = candidate.resolve()
    if target != base and base not in target.parents:
        raise SandboxError(f"'{rel_path}' resolves outside the selected workspace folder")
    return target


def _skip(p: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in p.parts) or p.suffix.lower() in SKIP_SUFFIXES


def tool_list_dir(workspace_root: str, path: str = ".") -> dict:
    try:
        p = safe_path(workspace_root, path)
    except SandboxError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"'{path}' does not exist"}
    if not p.is_dir():
        return {"error": f"'{path}' is not a directory"}
    lines = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if c.name in SKIP_DIR_NAMES:
                continue
            lines.append(("DIR  " if c.is_dir() else "FILE ") + c.name)
    except PermissionError:
        return {"error": f"permission denied reading '{path}'"}
    return {"result": "\n".join(lines) if lines else "(empty directory)"}


def tool_read_file(workspace_root: str, path: str) -> dict:
    try:
        p = safe_path(workspace_root, path)
    except SandboxError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"'{path}' does not exist"}
    if not p.is_file():
        return {"error": f"'{path}' is not a file"}
    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": f"could not read '{path}': {e}"}
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n\n... [truncated; file is {len(text)} characters total]"
    return {"result": text}


def tool_search_files(workspace_root: str, query: str, path: str = ".") -> dict:
    try:
        p = safe_path(workspace_root, path)
    except SandboxError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"'{path}' does not exist"}
    matches = []
    q = query.lower()
    try:
        for f in p.rglob("*"):
            if len(matches) >= MAX_SEARCH_MATCHES:
                break
            if not f.is_file() or _skip(f):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if q in line.lower():
                        rel = f.relative_to(Path(workspace_root).resolve())
                        matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            break
            except Exception:
                continue
    except PermissionError:
        return {"error": f"permission denied searching '{path}'"}
    if not matches:
        return {"result": "no matches"}
    suffix = f"\n... (capped at {MAX_SEARCH_MATCHES} matches)" if len(matches) >= MAX_SEARCH_MATCHES else ""
    return {"result": "\n".join(matches) + suffix}


def tool_write_file_stage(workspace_root: str, path: str, content: str) -> dict:
    """Stage a write for review -- never touches disk. Returns the diff and everything the UI
    needs to show an approve/reject prompt and, on approval, call apply_write with the same path
    and content."""
    try:
        p = safe_path(workspace_root, path)
    except SandboxError as e:
        return {"error": str(e)}
    is_new = not p.exists()
    old_text = "" if is_new else p.read_text(errors="replace")
    diff = "\n".join(difflib.unified_diff(
        old_text.splitlines(), content.splitlines(),
        fromfile=("/dev/null" if is_new else path), tofile=path, lineterm="",
    ))
    if not diff:
        diff = "(no changes -- new content is identical to the current file)"
    return {
        "result": "Change staged for review; not written to disk yet.",
        "diff": diff,
        "path": path,
        "content": content,
        "is_new": is_new,
    }


def tool_apply_write(workspace_root: str, path: str, content: str) -> dict:
    """Actually writes to disk. Only ever called from the UI's Approve action, never directly
    from a model tool call."""
    try:
        p = safe_path(workspace_root, path)
    except SandboxError as e:
        return {"error": str(e)}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as e:
        return {"error": f"failed to write '{path}': {e}"}
    return {"result": f"wrote {len(content)} characters to '{path}'"}


DISPATCH = {
    "list_dir": lambda root, args: tool_list_dir(root, args.get("path", ".")),
    "read_file": lambda root, args: tool_read_file(root, args.get("path", "")),
    "search_files": lambda root, args: tool_search_files(root, args.get("query", ""), args.get("path", ".")),
    "write_file": lambda root, args: tool_write_file_stage(root, args.get("path", ""), args.get("content", "")),
}
