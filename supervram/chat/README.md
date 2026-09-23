# SuperVRAM chat

A local chat UI for the models downloaded in this project, with a picker between all of them, an optional
**Tiered (SuperVRAM)** mode that runs this project's own VRAM+RAM+SSD expert cache instead of a plain load, and a
reasoning-effort control. No third-party dependencies (Python stdlib backend + a static HTML/JS page); nothing leaves
this machine.

## Run it

```bash
python3 chat/server.py --port 8788
```

or, inside this app, `preview_start` with name `svram-chat` (configured in `../../.claude/launch.json`). Then open
http://localhost:8788.

## How it's built

Two processes:

- **`chat/server.py`** (port 8788, this repo's code): lists the `.gguf` files under `~/models` (or `$SUPERVRAM_MODELS`),
  and — since only one large model fits in 24 GB of VRAM at a time — manages a single `llama-server` child process,
  stopping and restarting it whenever a different model or mode is picked. It does **not** proxy chat traffic; it only
  answers `GET /api/models`, `GET /api/status`, and `POST /api/select {model_id, mode}`.
- **`llama-server`** (port 8090, upstream llama.cpp, built as part of this project's normal CUDA build): the real
  OpenAI-compatible inference backend. The frontend talks to it **directly** (its CORS is open by default) for
  `/v1/chat/completions`, so streaming just works without `chat/server.py` needing to relay bytes.

`chat/index.html` is the whole frontend: model list, mode toggle, reasoning controls, message list, and a `fetch` +
`ReadableStream` reader that parses the OpenAI-style `data: {...}` SSE stream itself.

### Model launch and automatic fallback

For MoE models (name contains `A3B`), two launch recipes:

| Mode | What it runs |
|---|---|
| Fast | `-ngl 999` (whole model on GPU) as a first try |
| **Tiered** | `-ngl 999 --moe-expert-storage cache --moe-expert-cache-size 14336 --moe-expert-direct-io` — this project's classic expert slot cache, the same flags used throughout `results/overnight/` |

For the dense model(s), just `-ngl 999`.

Since not every model fits at the first attempt on this GPU (e.g. a 29 GB Q8_0 file alone exceeds 24 GB of VRAM), each
mode has a short fallback ladder (in `chat/server.py`: `MOE_FAST_LADDER`, `MOE_TIERED_LADDER`, `DENSE_LADDER`) that
tries progressively more CPU/cache offload until one boots without an early crash (the signature of running out of
VRAM). The UI's loading overlay shows which configuration it's currently trying.

### Reasoning control

`llama-server --reasoning-format deepseek` puts the model's thinking trace in a separate `message.reasoning_content`
field instead of leaving `<think>` tags inline, so the frontend can render it as a collapsed, expandable section above
each answer without parsing tags itself. The effort level (`low`/`medium`/`high`/`xhigh`) is sent per-request as the
OpenAI-compatible `reasoning_effort` field in the chat-completions body — a real `llama-server` parameter passed to
the model's own chat template, not a UI-side prompt hack — and `reasoning_effort: "none"` when the Thinking toggle is
off. This is still a **soft, model-level control**: the model decides how much to actually think, not a hard token cap.

## A bug found while building this, worth knowing about

Rapid, overlapping `POST /api/select` calls (e.g. clicking a model card and then the mode toggle within about a
second) used to corrupt the result: two background threads raced over the same `llama-server` child-process handle,
each one's `stop_current()` capable of killing the other's just-started process — every attempt looked like it failed
even when the first one actually would have worked. Fixed with a monotonically increasing "generation" counter
(`bump_gen`/`is_current` in `chat/server.py`): a new `/api/select` call always wins, and a superseded request's
background thread notices within about a second (it polls `is_current()` in its own health-check loop, not just once
at the start) and gives up cleanly instead of fighting over the process or the reported status.

## What this does not do

- **No image generation.** None of the downloaded models are diffusion/image models; they're text-generation LLMs.
- **No image understanding either, yet.** The uncensored dense model's upstream repo ships a separate
  `mmproj-*.gguf` vision-adapter file that would enable it, but that file has not been downloaded — the chat UI has
  no vision input today. Downloading it and wiring `--mmproj` into `chat/server.py`'s launch command would be a
  follow-up, not a large change.
- **Context is fixed at 8192 tokens** (`CTX_SIZE` in `chat/server.py`) — a deliberate, conservative default so more
  models fit without hand-tuning per model; the SuperVRAM caches used throughout the rest of this project (32k
  context) are not exercised by chat traffic through this UI as configured.

## Where everything lives

`chat/server.py` (orchestrator), `chat/index.html` (frontend), log of the currently-running `llama-server`:
`/tmp/claude-1000/chat-llama-server.log`. Models come from `~/models/*.gguf` — see `../results/overnight/` for how
each one was downloaded and what it's for.
