#!/usr/bin/env python3
"""Build the human-written prose corpus used for the ground-truth perplexity check: llama.cpp's docs/*.md with code blocks and
tables removed. Usage: make_corpus.py [llama.cpp dir] [output file]  (defaults: ../third_party/llama.cpp, /tmp/corpus.txt)"""
import glob, os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "third_party", "llama.cpp")
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/corpus.txt"
out = []
for f in sorted(glob.glob(os.path.join(root, "docs/**/*.md"), recursive=True)) + [os.path.join(root, "README.md"), os.path.join(root, "CONTRIBUTING.md")]:
    try:
        t = open(f, encoding="utf-8").read()
    except FileNotFoundError:
        continue
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    lines = [l for l in t.splitlines() if not l.lstrip().startswith(("|", "<", "![", "[!", "#include", "$ ", "    "))]
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(t) > 2000:
        out.append(t)
text = "\n\n".join(out)
open(out_path, "w").write(text)
print(len(out), "files,", len(text), "chars ->", out_path)
