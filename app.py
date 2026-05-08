"""
Zava Arena — On-Device AI for Live Events
=========================================
A Flask application showcasing on-device AI for a modern live-events arena.
Demonstrates per-persona AI assistants
running entirely on Copilot+ PC NPUs via Microsoft Foundry Local +
ONNX Runtime DirectML — no cloud, no data exfil, sub-second latency
for the millions of micro-interactions a live event generates.

Tabs:
  1. Home                  – Overview of on-device AI for live events
  2. Concertgoer Concierge – Wayfinding, set lists, food, accessibility
  3. Performer Coach       – Vocal/energy coaching, crowd insights, setlist tips
  4. Operations Playbook   – Crowd flow, incident response, equipment triage
  5. Content Authenticity  – Detect AI-generated text/image/video on the NPU
  6. NPU Dashboard         – Live metrics, cost savings, offline proof
"""

import os
import sys
import json
import os
import sys
import json
import time
import uuid
import subprocess
import traceback
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Silicon detection — ARM64 / Snapdragon-aware
# ---------------------------------------------------------------------------
SILICON = "unknown"

def _detect_silicon() -> str:
    """Detect CPU silicon family via WMI (platform.machine() lies under x64 emulation on ARM64)."""
    global SILICON
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=5,
        )
        cpu = result.stdout.strip().lower()
        if "qualcomm" in cpu or "snapdragon" in cpu:
            SILICON = "qualcomm"
        elif "intel" in cpu:
            SILICON = "intel"
        elif "amd" in cpu:
            SILICON = "amd"
        else:
            SILICON = "unknown"
    except Exception:
        SILICON = "unknown"
    return SILICON

_detect_silicon()
print(f"[STARTUP] Silicon detected: {SILICON}")

# ---------------------------------------------------------------------------
# Foundry Local bootstrap (NPU pre-loaded via HTTP, CPU fallback)
# Optimized for Snapdragon X / QNN runtime
# ---------------------------------------------------------------------------
foundry_ok = False
model_id = None
foundry_service_url = None
npu_alias = None
use_npu = False
hardware_label = "CPU"  # "NPU", "GPU", or "CPU" — updated by init_foundry()

# Preferred model family — Phi-4-mini is our go-to across NPU / GPU / CPU.
# On Intel (OpenVINO), the runtime tries each tier in order and falls back
# automatically if the NPU variant fails to compile on the local NPU driver.
PHI4_FAMILY = "phi-4-mini"

# NPU aliases to try first (most-preferred → least-preferred).
NPU_ALIAS_PREFERENCE = [
    "phi-4-mini",         # Intel OpenVINO NPU / Snapdragon QNN — primary
    "phi-3-mini-4k",      # Phi family fallback
    "qwen2.5-1.5b",       # tiny, fast — last NPU resort
]

# When an Intel NPU compile fails we must keep the user on Phi-4-mini if
# possible — pre-cache the GPU & generic-GPU variants so they appear in /v1/models.
INTEL_GPU_PREFERENCE = [
    "phi-4-mini-instruct-openvino-gpu:2",   # OpenVINO GPU, same Phi-4-mini weights
    "Phi-4-mini-instruct-generic-gpu:5",    # generic GPU fallback
]

# CPU model IDs to try via HTTP — first match wins
CPU_MODEL_PREFERENCE = [
    "Phi-4-mini-instruct-generic-cpu",
    "Phi-3.5-mini-instruct-generic-cpu",
    "qwen2.5-0.5b-instruct-generic-cpu",
]

# Reason the runtime fell back from NPU (surfaced in /api/status for UI).
fallback_reason = ""


def _discover_foundry_port() -> str | None:
    """Parse `foundry service status` to get the running service URL."""
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "http://" in line:
                import re
                m = re.search(r"(https?://[\d.]+:\d+)", line)
                if m:
                    return m.group(1)
    except Exception as exc:
        print(f"[STARTUP] foundry CLI not available: {exc}")
    return None


def _detect_npu_alias() -> str | None:
    """Find the best cached NPU model alias via `foundry model list`."""
    try:
        result = subprocess.run(
            ["foundry", "model", "list"],
            capture_output=True, text=True, timeout=15,
        )
        print(f"[STARTUP] foundry model list output:\n{result.stdout[:600]}")
        npu_aliases = set()
        all_aliases = set()
        current_alias = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            # Detect device tags on same line or subsequent lines
            if parts[0] in ("CPU", "NPU", "GPU", "Auto", "QNN", "DirectML"):
                if parts[0] in ("NPU", "QNN") and current_alias:
                    npu_aliases.add(current_alias)
            elif not parts[0].startswith("-") and not parts[0].startswith("Alias") and not parts[0].startswith("="):
                current_alias = parts[0]
                all_aliases.add(current_alias)
                # Check if device info is on the same line
                line_upper = line.upper()
                if "NPU" in line_upper or "QNN" in line_upper:
                    npu_aliases.add(current_alias)

        print(f"[STARTUP] All aliases found: {all_aliases}")
        print(f"[STARTUP] NPU/QNN aliases found: {npu_aliases}")

        for pref in NPU_ALIAS_PREFERENCE:
            if pref in npu_aliases:
                return pref
        # If no NPU aliases detected, but preferred aliases exist, return them
        # (let foundry model load decide the device)
        if not npu_aliases:
            for pref in NPU_ALIAS_PREFERENCE:
                if pref in all_aliases:
                    print(f"[STARTUP] No explicit NPU tag, but alias '{pref}' available — will attempt NPU load")
                    return pref
        return next(iter(npu_aliases), None)
    except Exception as exc:
        print(f"[STARTUP] Could not detect NPU models: {exc}")
        return None


def _foundry_get(path: str, timeout: int = 10):
    """GET helper returning parsed JSON, or None on failure."""
    try:
        resp = urllib.request.urlopen(f"{foundry_service_url}{path}", timeout=timeout)
        return json.loads(resp.read())
    except Exception:
        return None


def _foundry_post(path: str, body: dict, timeout: int = 120):
    """POST helper returning parsed JSON, or None on failure."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{foundry_service_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def init_foundry():
    """Discover running Foundry Local service — prefer NPU (OpenVINO/QNN), fall back to GPU then CPU."""
    global foundry_ok, model_id, foundry_service_url, npu_alias, use_npu, hardware_label, fallback_reason

    foundry_ok = False
    model_id = None
    foundry_service_url = None
    npu_alias = None
    use_npu = False
    hardware_label = "CPU"
    fallback_reason = ""

    # 1. Ensure Foundry HTTP service is running
    service_url = _discover_foundry_port()
    if not service_url:
        try:
            subprocess.run(["foundry", "service", "start"],
                           capture_output=True, text=True, timeout=30)
            service_url = _discover_foundry_port()
        except Exception:
            pass

    if not service_url:
        print("[STARTUP] Foundry Local service not running. UI-preview mode.")
        return

    foundry_service_url = service_url
    print(f"[STARTUP] Foundry Local HTTP service at {service_url}")

    # ---- Enumerate all NPU-capable aliases from the Foundry catalog ----
    def _list_catalog_npu_aliases():
        """Return list of (alias, model_id) for every NPU variant Foundry knows about."""
        try:
            result = subprocess.run(
                ["foundry", "model", "list"],
                capture_output=True, text=True, timeout=30,
            )
            out = []
            current_alias = None
            for line in result.stdout.splitlines():
                # Lines look like:  "phi-4-mini   NPU   chat, tools   2.15 GB   MIT   phi-4-mini-instruct-openvino-npu:3"
                # Or continuation: "             NPU   chat ..."
                stripped = line.strip()
                if not stripped or stripped.startswith("-") or stripped.startswith("Alias"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                # If first column is non-blank in original line, it's a new alias
                if not line.startswith(" ") and not line.startswith("\t"):
                    current_alias = parts[0]
                # Find a model-id token (contains ':' and 'npu')
                for tok in parts:
                    if ":" in tok and "npu" in tok.lower() and current_alias:
                        out.append((current_alias, tok))
                        break
            return out
        except Exception as exc:
            print(f"[STARTUP] Catalog enumeration failed: {exc}")
            return []

    # Prioritize: download every NPU variant in the catalog (Phi-4-mini first),
    # so we can probe each in turn.
    npu_catalog = _list_catalog_npu_aliases()
    if npu_catalog:
        # Sort: phi-4-mini family first, then phi-3, then phi, then others
        def _cat_rank(item):
            mid = item[1].lower()
            if "phi-4-mini" in mid: return 0
            if "phi-3" in mid: return 1
            if "phi" in mid: return 2
            if "qwen2.5-1.5b" in mid: return 3
            if "qwen2.5-0.5b" in mid: return 4
            return 5
        npu_catalog.sort(key=_cat_rank)
        print(f"[STARTUP] NPU catalog (priority order): {[m for _, m in npu_catalog]}")

        # Get currently cached models so we don't re-download
        try:
            cached = subprocess.run(
                ["foundry", "cache", "list"],
                capture_output=True, text=True, timeout=15,
            ).stdout.lower()
        except Exception:
            cached = ""

        # Download up to 4 NPU variants we don't have yet (cap to keep startup fast)
        to_download = []
        for _, mid in npu_catalog[:6]:
            if mid.split(":")[0].lower() not in cached:
                to_download.append(mid)
        for mid in to_download[:4]:
            print(f"[STARTUP] Downloading NPU variant for prioritization: {mid}")
            try:
                subprocess.run(
                    ["foundry", "model", "download", mid],
                    capture_output=True, text=True, timeout=600,
                )
            except Exception as exc:
                print(f"[STARTUP] Download skipped ({mid}): {exc}")

    # Pre-cache Phi-4-mini GPU as a safety-net if Intel NPU compile fails
    if SILICON == "intel":
        try:
            cached2 = subprocess.run(
                ["foundry", "cache", "list"],
                capture_output=True, text=True, timeout=15,
            ).stdout.lower()
            if "phi-4-mini-instruct-openvino-gpu" not in cached2:
                print("[STARTUP] Pre-caching phi-4-mini-instruct-openvino-gpu:2 (NPU fallback)")
                subprocess.run(
                    ["foundry", "model", "download", "phi-4-mini-instruct-openvino-gpu:2"],
                    capture_output=True, text=True, timeout=600,
                )
        except Exception as exc:
            print(f"[STARTUP] GPU pre-cache check failed: {exc}")

    # 2. List every model the HTTP service knows about, then probe NPU first.
    models_data = _foundry_get("/v1/models")
    if models_data and "data" in models_data:
        available_ids = [m["id"] for m in models_data["data"]]
        print(f"[STARTUP] Available HTTP models: {available_ids}")

        def _device_rank(mid: str) -> int:
            m = mid.lower()
            if any(t in m for t in ("npu", "qnn", "directml", "qualcomm")):
                return 0
            if "gpu" in m:
                return 1
            return 2  # CPU / unknown

        def _family_rank(mid: str) -> int:
            """Lower = preferred. Phi-4-mini family wins, OpenVINO beats generic on Intel."""
            m = mid.lower()
            phi4 = "phi-4-mini" in m
            phi3 = "phi-3" in m
            phi = "phi" in m
            openvino = "openvino" in m
            base = 0 if phi4 else (2 if phi3 else (4 if phi else 6))
            return base + (0 if openvino else 1)

        # Sort by (device tier, family tier) so Phi-4-mini NPU wins, then any other NPU,
        # then Phi-4-mini GPU, then any other GPU, then CPU.
        candidates = sorted(available_ids, key=lambda x: (_device_rank(x), _family_rank(x)))
        print(f"[STARTUP] Probe order (NPU prioritized): {candidates}")

        def _probe_model(mid: str, probe_timeout: int = 180) -> bool:
            """Send a tiny chat completion to verify the model actually runs.
            Use a generous timeout — first compile on cold NPU/GPU can take 60-120s."""
            global fallback_reason
            try:
                _foundry_post("/v1/chat/completions", {
                    "model": mid,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                }, timeout=probe_timeout)
                return True
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="ignore")[:600]
                except Exception:
                    pass
                print(f"[STARTUP] Probe failed [{mid}] HTTP {e.code}: {body[:240]}")
                if _device_rank(mid) == 0 and not fallback_reason:
                    if "Failed to compile" in body or "intel_npu" in body:
                        fallback_reason = (
                            f"Intel NPU could not compile {mid.split(':')[0]} "
                            f"(OpenVINO plugin error). Falling back to GPU."
                        )
                    else:
                        fallback_reason = f"NPU probe failed for {mid.split(':')[0]}: HTTP {e.code}"
                return False
            except Exception as exc:
                print(f"[STARTUP] Probe failed [{mid}]: {exc}")
                return False

        for mid in candidates:
            tier = _device_rank(mid)
            tier_name = "NPU" if tier == 0 else ("GPU" if tier == 1 else "CPU")
            # NPU/GPU cold compile can take 1-3 minutes on first call
            timeout = 240 if tier <= 1 else 60
            print(f"[STARTUP] Probing {tier_name} model: {mid} (timeout={timeout}s) ...")
            if _probe_model(mid, probe_timeout=timeout):
                model_id = mid
                use_npu = (tier == 0)
                foundry_ok = True
                hardware_label = tier_name
                print(f"[STARTUP] ✓ Model verified: {model_id}")
                print(f"[STARTUP] ✓ Running on: {tier_name} via HTTP service")
                return
            else:
                print(f"[STARTUP] Skipping {mid} — trying next candidate")

        print("[STARTUP] No pre-loaded model passed verification; will try to load one.")

    # 3. Try to pre-load an NPU model into the service
    detected = _detect_npu_alias()
    if detected:
        # Try multiple device flags — Snapdragon uses QNN, Intel uses NPU
        device_attempts = ["NPU"]
        if SILICON == "qualcomm":
            device_attempts = ["QNN", "NPU"]

        loaded = False
        for device_flag in device_attempts:
            print(f"[STARTUP] Pre-loading NPU model '{detected}' via foundry model load --device {device_flag} ...")
            try:
                result = subprocess.run(
                    ["foundry", "model", "load", detected, "--device", device_flag],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    loaded = True
                    break
                else:
                    print(f"[STARTUP] foundry model load --device {device_flag} failed (rc={result.returncode}): {result.stderr[:300]}")
            except subprocess.TimeoutExpired:
                print(f"[STARTUP] foundry model load --device {device_flag} timed out (120s)")
            except Exception as exc:
                print(f"[STARTUP] foundry model load error: {exc}")

        # Also try without --device flag as fallback
        if not loaded:
            print(f"[STARTUP] Trying foundry model load '{detected}' without --device flag ...")
            try:
                result = subprocess.run(
                    ["foundry", "model", "load", detected],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    loaded = True
                else:
                    print(f"[STARTUP] foundry model load (no device) failed (rc={result.returncode}): {result.stderr[:300]}")
            except Exception as exc:
                print(f"[STARTUP] foundry model load error: {exc}")

        if loaded:
            time.sleep(2)
            models_data = _foundry_get("/v1/models")
            if models_data and "data" in models_data:
                npu_ids = [m["id"] for m in models_data["data"]
                           if any(tag in m["id"].lower() for tag in ("npu", "qnn", "directml", "qualcomm"))]
                best = None
                for mid in npu_ids:
                    if detected.replace("-", "") in mid.replace("-", "").lower():
                        best = mid
                        break
                if not best and npu_ids:
                    best = npu_ids[0]
                if best:
                    model_id = best
                    npu_alias = detected
                    use_npu = True
                    foundry_ok = True
                    hardware_label = "NPU"
                    print(f"[STARTUP] NPU model ready via HTTP: {model_id}")
                    return
                else:
                    # Model loaded but no NPU tag — check if any new model appeared
                    all_ids = [m["id"] for m in models_data["data"]]
                    for mid in all_ids:
                        if detected.replace("-", "") in mid.replace("-", "").lower():
                            model_id = mid
                            npu_alias = detected
                            use_npu = True
                            foundry_ok = True
                            hardware_label = "NPU"
                            print(f"[STARTUP] Model loaded (assuming NPU): {model_id}")
                            return

    # 4. Fall back to CPU via HTTP service
    models_data = _foundry_get("/v1/models")
    if not models_data or "data" not in models_data:
        print("[STARTUP] Could not list models. UI-preview mode.")
        foundry_service_url = None
        return

    available_ids = [m["id"] for m in models_data["data"]]

    # Strict CPU match — never accidentally label an NPU/GPU variant as "CPU"
    cpu_only = [m for m in available_ids
                if "cpu" in m.lower()
                and "npu" not in m.lower()
                and "gpu" not in m.lower()]

    for pref in CPU_MODEL_PREFERENCE:
        pref_lower = pref.lower()
        for mid in cpu_only:
            if mid.lower().startswith(pref_lower):
                model_id = mid
                break
        if model_id:
            break

    if not model_id and cpu_only:
        model_id = cpu_only[0]

    if not model_id:
        # Last resort — keep service alive but be honest about the device
        if available_ids:
            model_id = available_ids[0]
            print(f"[STARTUP] No CPU-only model available; using {model_id} (may be unstable)")
            hardware_label = "CPU"  # honest default; we couldn't probe
        else:
            print("[STARTUP] No models available. UI-preview mode.")
            foundry_service_url = None
            return
    else:
        hardware_label = "CPU"

    foundry_ok = True
    print(f"[STARTUP] Selected CPU model: {model_id}")
    print(f"[STARTUP] Running on: CPU (HTTP service)")

init_foundry()

# Snapdragon: skip warmup — QNN runtime destabilizes with rapid reconnection
if foundry_ok and SILICON != "qualcomm":
    try:
        print("[STARTUP] Warming up model (Intel/AMD — safe to warmup)...")
        _warmup_body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 8,
        }
        _foundry_post("/v1/chat/completions", _warmup_body, timeout=60)
        print("[STARTUP] Warmup complete — model is hot.")
    except Exception as exc:
        print(f"[STARTUP] Warmup skipped: {exc}")
elif SILICON == "qualcomm":
    print("[STARTUP] Snapdragon detected — skipping warmup (QNN loads on first request)")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve previously-uploaded Ops photos. Stays local — no CDN, no cloud."""
    return send_from_directory(str(UPLOAD_DIR), filename)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
inference_log: list[dict] = []


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _run_inference(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> dict:
    """Run a chat completion via Foundry HTTP service (NPU or CPU). Log metrics."""
    if not foundry_ok or not foundry_service_url or not model_id:
        return {
            "response": "[Demo mode — Foundry Local not connected. Install & start Foundry Local to enable on-device AI.]",
            "text": "[Demo mode — Foundry Local not connected. Install & start Foundry Local to enable on-device AI.]",
            "tokens": 0,
            "latency_ms": 0,
            "cloud_cost_saved": "$0.00",
            "hardware": hardware_label,
        }

    # Many OpenVINO NPU model builds cap output at 528 tokens. Cap requests
    # at a safe value to avoid 500 errors from oversized max_tokens.
    safe_max = min(max_tokens, 480)

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": safe_max,
    }

    hardware = hardware_label
    t0 = time.perf_counter()
    try:
        result = _foundry_post("/v1/chat/completions", body, timeout=120)
    except Exception as exc:
        print(f"[INFERENCE] HTTP call failed ({hardware}): {exc}")
        # Reconnect & retry — Foundry port may have rotated, or the chosen
        # model may have failed (e.g. Intel NPU compile error). init_foundry()
        # will re-probe and pick a working model (NPU → GPU → CPU).
        try:
            init_foundry()
            body["model"] = model_id
            result = _foundry_post("/v1/chat/completions", body, timeout=120)
        except Exception as exc2:
            return {
                "response": f"[Error: Could not reach Foundry Local — {exc2}]",
                "text": f"[Error: Could not reach Foundry Local — {exc2}]",
                "tokens": 0, "latency_ms": 0, "cloud_cost_saved": "$0.00",
                "hardware": hardware_label,
            }

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    text = ""
    choices = result.get("choices", [])
    if choices:
        choice = choices[0]
        msg = choice.get("message") or choice.get("delta") or {}
        text = msg.get("content", "")

    usage = result.get("usage") or {}
    total_tokens = usage.get("total_tokens", 0)
    if not total_tokens:
        total_tokens = _estimate_tokens(system_prompt + user_prompt) + _estimate_tokens(text)

    est_cost = round(total_tokens * 0.00001, 6)

    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "tokens": total_tokens,
        "latency_ms": elapsed_ms,
        "cloud_cost_saved": f"${est_cost:.4f}",
        "hardware": hardware,
    }
    inference_log.append(entry)

    return {
        "response": text,
        "text": text,  # legacy alias
        "tokens": total_tokens,
        "latency_ms": elapsed_ms,
        "cloud_cost_saved": f"${est_cost:.4f}",
        "hardware": hardware,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return NPU / Foundry Local availability + silicon info."""
    runtime = "OpenVINO" if SILICON == "intel" else ("QNN" if SILICON == "qualcomm" else "Foundry Local")
    short_model = (model_id or "N/A").split(":")[0]
    return jsonify({
        "ready": foundry_ok,
        "foundry_connected": foundry_ok,
        "model": short_model,
        "model_full": model_id or "N/A",
        "endpoint": foundry_service_url or "N/A",
        "mode": (f"on-device {hardware_label}" if foundry_ok else "UI preview (no AI)"),
        "hardware": (hardware_label if foundry_ok else "none"),
        "runtime": runtime,
        "silicon": SILICON,
        "fallback_reason": fallback_reason,
        "message": (
            f"On-device {hardware_label} ready · {short_model}"
            if foundry_ok else "NPU initializing…"
        ),
    })
@app.route("/api/performer", methods=["POST"])
def api_performer():
    """Performer Coach — vocal warmups, energy/setlist guidance for artists."""
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    system = (
        "You are the Performer Coach inside Zava Arena — an AI assistant for touring artists, "
        "vocalists, and band leaders performing in a modern 18,000-seat arena. "
        "Help with: vocal warmup routines, breath control, setlist sequencing for energy arcs, "
        "between-song banter ideas, crowd-energy reading, in-ear monitor tips, "
        "and how to use the arena's main LED wall + house PA to enhance moments. "
        "Be concise, actionable, and respectful of the artist's craft. Keep responses under 200 words."
    )
    result = _run_inference(system, user_msg, max_tokens=350)
    return jsonify(result)


@app.route("/api/concierge", methods=["POST"])
def api_concierge():
    """Concertgoer Concierge — fan-facing AI for set lists, wayfinding, accessibility."""
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    system = (
        "You are the Zava Arena Concertgoer Concierge — a friendly, on-device AI assistant for fans "
        "attending shows at Zava Arena, a modern 18,000-seat live-events venue with a large center-hung "
        "LED scoreboard, a main stage LED wall, and full house PA coverage. "
        "Help with: wayfinding (entrances, sections, restrooms, concessions), tonight's set list and "
        "artist trivia, accessibility services (companion seating, hearing loops, sensory accommodations, "
        "ASL interpreters), nearby food, rideshare pickup zones, lost & found, merch tips, and what to expect "
        "from the arena experience. Be warm, concise, and enthusiastic. If you don't know a real-time fact "
        "(e.g., tonight's exact set list), suggest checking the official Zava Arena app. Keep replies under 180 words."
    )
    result = _run_inference(system, user_msg, max_tokens=350)
    return jsonify(result)


@app.route("/api/ops", methods=["POST"])
def api_ops():
    """Operations Playbook — analyze briefings/incidents and produce playbooks for venue staff.

    Accepts optional fields:
      - scene: pre-canned scene description for sample stadium photos
      - image_filename: name of an uploaded photo (echoed back in the prompt)
      - text: typed or voice-dictated notes (Web Speech API runs on-device)
    """
    data = request.get_json(force=True)
    doc_text = (data.get("text") or "").strip()
    task = data.get("task", "playbook")
    scene = (data.get("scene") or "").strip()
    image_filename = (data.get("image_filename") or "").strip()

    if not doc_text and not scene and not image_filename:
        return jsonify({"error": "No briefing, voice notes, or photo provided"}), 400

    task_prompts = {
        "playbook": (
            "Produce a concise incident-response playbook from this Zava Arena briefing. "
            "Output: 1) Severity classification, 2) Immediate actions (first 60 seconds), "
            "3) Comms tree (who to notify), 4) De-escalation/mitigation steps, "
            "5) Post-event follow-up. Be terse, numbered, and actionable."
        ),
        "crowd": (
            "You are analyzing a crowd-flow briefing for Zava Arena. Identify: "
            "1) Bottleneck risks, 2) Sectioned routing recommendations, 3) Staffing reallocation, "
            "4) Egress timing windows, 5) Accessibility considerations. Format as a bulleted brief."
        ),
        "equipment": (
            "Triage this equipment/technical issue at Zava Arena. Output: "
            "1) Probable root cause, 2) Diagnostic steps in order, 3) Likely fix or workaround, "
            "4) Whether show can continue / requires hold, 5) Vendor escalation contact category."
        ),
    }

    system = "Zava Arena Operations AI. " + task_prompts.get(task, task_prompts["playbook"])

    # Compose a single user prompt from photo scene + voice/typed notes.
    # Phi-4-mini is text-only on the Intel NPU; the scene description fills in
    # the visual context the same way a vision-language model would.
    parts = []
    if scene:
        parts.append(f"Photo analysis (auto-extracted on-device): {scene}")
    elif image_filename:
        parts.append(f"Photo attached: {image_filename} (no auto-description available — rely on the notes below).")
    if doc_text:
        parts.append(f"Staff notes: {doc_text}")

    user_prompt = "\n\n".join(parts)[:1800]
    result = _run_inference(system, user_prompt, max_tokens=400)
    return jsonify(result)


@app.route("/api/upload-ops-image", methods=["POST"])
def api_upload_ops_image():
    """Save an Ops photo upload to the local uploads dir. Stays on this device."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file"}), 400

    allowed = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"}
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"File type .{ext} not allowed"}), 400

    fname = secure_filename(f"{uuid.uuid4().hex[:8]}_{f.filename}")
    save_path = UPLOAD_DIR / fname
    f.save(str(save_path))
    return jsonify({"filename": fname, "url": f"/uploads/{fname}"})


# ---------------------------------------------------------------------------
# Real-time multilingual captions
# ---------------------------------------------------------------------------
SUPPORTED_CAPTION_LANGS = {
    "es": "Spanish",
    "pt": "Portuguese (Brazil)",
    "fr": "French",
    "de": "German",
    "zh": "Mandarin Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "it": "Italian",
    "ru": "Russian",
    "tl": "Tagalog",
}


@app.route("/api/captions", methods=["POST"])
def api_captions():
    """Translate live artist banter into multiple languages in one NPU call.

    Body: { "text": str, "langs": ["es","pt",...] }

    Single Phi-4-mini call returns all translations as JSON so the NPU
    (which is single-request) doesn't have to queue per-language.
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    langs = data.get("langs") or ["es", "pt", "fr", "de", "zh", "ja"]

    if not text:
        return jsonify({"error": "No text to translate"}), 400

    # Filter to supported and de-dupe while preserving order
    seen = set()
    target_codes = []
    for code in langs:
        if code in SUPPORTED_CAPTION_LANGS and code not in seen:
            seen.add(code)
            target_codes.append(code)
    if not target_codes:
        return jsonify({"error": "No supported languages requested"}), 400

    label_lines = "\n".join(
        f'  "{code}": "{SUPPORTED_CAPTION_LANGS[code]} translation here"'
        for code in target_codes
    )

    system = (
        "You are a live-event caption translator for a stadium. "
        "Translate the speaker's line into every requested language. "
        "Keep the tone, register, and humor; never editorialize. "
        "Output a single valid JSON object whose keys are language codes "
        "and values are the translated string. No prose, no markdown, "
        "no code fences, no commentary — JSON only."
    )
    user_prompt = (
        f"Speaker said (English): {text[:600]}\n\n"
        f"Return JSON exactly in this shape:\n{{\n{label_lines}\n}}"
    )

    result = _run_inference(system, user_prompt, max_tokens=480)
    raw = result.get("response") or result.get("text") or ""

    # Try to parse JSON out of the response
    translations = {}
    parse_error = None
    if raw:
        candidate = raw.strip()
        # strip ```json fences if present
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
        # find first { and last }
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first >= 0 and last > first:
            candidate = candidate[first:last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                translations = {k: v for k, v in parsed.items()
                                if k in SUPPORTED_CAPTION_LANGS and isinstance(v, str)}
        except Exception as exc:
            parse_error = str(exc)

    return jsonify({
        "source": text,
        "translations": translations,
        "labels": {c: SUPPORTED_CAPTION_LANGS[c] for c in target_codes},
        "raw_response": raw if not translations else None,
        "parse_error": parse_error,
        "tokens": result.get("tokens"),
        "latency_ms": result.get("latency_ms"),
        "hardware": result.get("hardware"),
        "cloud_cost_saved": result.get("cloud_cost_saved"),
    })


@app.route("/api/captions/languages")
def api_captions_languages():
    return jsonify(SUPPORTED_CAPTION_LANGS)


# ---------------------------------------------------------------------------
# Crew shift handover brief
# ---------------------------------------------------------------------------
@app.route("/api/handover", methods=["POST"])
def api_handover():
    """Generate a structured handover brief for the incoming supervisor.

    Body: { "memo": str, "log": str (optional), "shift_role": str (optional) }
    Designed to run in elevators and stairwells where there is no Wi-Fi.
    """
    data = request.get_json(force=True)
    memo = (data.get("memo") or "").strip()
    log = (data.get("log") or "").strip()
    role = (data.get("shift_role") or "Floor Supervisor").strip() or "Floor Supervisor"

    if not memo and not log:
        return jsonify({"error": "Dictate a memo or paste an incident log first"}), 400

    system = (
        "You are the Zava Arena shift-handover AI. Convert the outgoing supervisor's "
        "voice memo and incident log into a tight, scannable handover brief for "
        "the incoming supervisor. Be terse — every section must fit on a phone "
        "screen. Output exactly these sections, in this order, using Markdown "
        "headings:\n"
        "## Snapshot — one-line state of the venue right now\n"
        "## Open issues — numbered, each one line, with location + owner\n"
        "## Watch items — things that may escalate this shift\n"
        "## Comms — who has been notified, who still needs to be\n"
        "## First 15 minutes — what the incoming supervisor should do first\n"
        "Never invent facts. If a section has no data, write 'None reported.'"
    )
    parts = [f"Outgoing role: {role}"]
    if memo:
        parts.append(f"Voice memo (transcribed on-device): {memo}")
    if log:
        parts.append(f"Incident log:\n{log}")
    user_prompt = "\n\n".join(parts)[:1800]

    result = _run_inference(system, user_prompt, max_tokens=450)
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    """Return cumulative inference metrics for the dashboard."""
    total_tokens = sum(e["tokens"] for e in inference_log)
    total_cost = sum(float(e["cloud_cost_saved"].replace("$", "")) for e in inference_log)
    avg_latency = (
        round(sum(e["latency_ms"] for e in inference_log) / len(inference_log))
        if inference_log else 0
    )
    return jsonify({
        "total_inferences": len(inference_log),
        "total_tokens": total_tokens,
        "total_cloud_cost_saved": f"${total_cost:.4f}",
        "avg_latency_ms": avg_latency,
        "log": inference_log[-20:],
    })


# ---------------------------------------------------------------------------
# Content Authenticity Detector (NPU via ONNX Runtime DirectML)
# ---------------------------------------------------------------------------
_content_analyzer = None
_content_analyzer_err = None


def _get_content_analyzer():
    """Lazy-load the NPU content analyzer on first use."""
    global _content_analyzer, _content_analyzer_err
    if _content_analyzer is not None or _content_analyzer_err is not None:
        return _content_analyzer
    try:
        from detector.analyzer import NPUContentAnalyzer
        _content_analyzer = NPUContentAnalyzer()
        print(f"[STARTUP] Content Detector ready — provider: {_content_analyzer.provider}")
    except Exception as e:
        _content_analyzer_err = str(e)
        print(f"[STARTUP] Content Detector unavailable: {e}")
    return _content_analyzer


@app.route("/api/detector/status")
def api_detector_status():
    a = _get_content_analyzer()
    if a is None:
        return jsonify({"available": False, "error": _content_analyzer_err})
    return jsonify({
        "available": True,
        "npu_available": a.npu_available,
        "execution_provider": a.provider,
        "device_name": a.device_name,
        "image_model": a.image_model_name,
        "text_model": a.text_model_name,
    })


@app.route("/api/detector/text", methods=["POST"])
def api_detector_text():
    a = _get_content_analyzer()
    if a is None:
        return jsonify({"error": f"Detector unavailable: {_content_analyzer_err}"}), 503
    text = (request.form.get("text") or (request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text input is required."}), 400
    if len(text) > 50_000:
        return jsonify({"error": "Text must be under 50,000 characters."}), 400
    try:
        t0 = time.perf_counter()
        result = a.analyze_text(text)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["provider"] = a.provider
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/api/detector/image", methods=["POST"])
def api_detector_image():
    a = _get_content_analyzer()
    if a is None:
        return jsonify({"error": f"Detector unavailable: {_content_analyzer_err}"}), 503
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Image file is required."}), 400
    contents = f.read()
    if len(contents) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400
    if len(contents) > 20 * 1024 * 1024:
        return jsonify({"error": "Image must be under 20 MB."}), 400
    try:
        t0 = time.perf_counter()
        result = a.analyze_image(contents)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["provider"] = a.provider
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400


@app.route("/api/detector/video", methods=["POST"])
def api_detector_video():
    import tempfile, shutil
    a = _get_content_analyzer()
    if a is None:
        return jsonify({"error": f"Detector unavailable: {_content_analyzer_err}"}), 503
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Video file is required."}), 400
    contents = f.read()
    if len(contents) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400
    if len(contents) > 100 * 1024 * 1024:
        return jsonify({"error": "Video must be under 100 MB."}), 400
    tmp_dir = tempfile.mkdtemp()
    ext = Path(f.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = os.path.join(tmp_dir, "upload" + ext)
    try:
        with open(tmp_path, "wb") as out:
            out.write(contents)
        t0 = time.perf_counter()
        result = a.analyze_video(tmp_path)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["provider"] = a.provider
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Could not process video: {exc}"}), 400
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    runtime_name = "OpenVINO" if SILICON == "intel" else ("QNN" if SILICON == "qualcomm" else "Foundry Local")
    print("  Zava Arena — On-Device AI for Live Events")
    print("  Powered by Microsoft Surface + Foundry Local")
    print(f"  Runtime: {runtime_name} ({SILICON.upper()} silicon)")
    print("=" * 60)
    print(f"  Silicon: {SILICON.upper()}")
    print(f"  Model loading may take a moment on first run...")
    print(f"  Once ready, open → http://localhost:5006\n")
    app.run(host="127.0.0.1", port=5006, debug=False)
