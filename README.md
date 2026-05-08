# Zava Arena — On-Device AI for Live Events

A Flask demo showcasing six on-device AI personas for a modern 18,000-seat
live-events arena. Everything runs locally on a Copilot+ PC NPU via
**Microsoft Foundry Local** and **OpenVINO** — no cloud round-trips.

Optimized for **Intel Core Ultra (AI Boost NPU)** with Phi-4-mini.
Snapdragon X / QNN runtime is also auto-detected.

## Personas

| Tab | Who it's for | Why on-device |
|---|---|---|
| **Concertgoer Concierge** | Fans / kiosks | Cell coverage collapses to zero when 18,000 phones hit one tower. The kiosk keeps working. |
| **Performer Coach** | Touring artist on Surface | Vocal/setlist/crowd coaching that never leaves the dressing room. |
| **Operations Playbook** | Venue staff (with photo + voice input) | Incident triage that never sends fan PII or scene photos off-prem. |
| **Live Captions** | Accessibility + multilingual fans | Real-time English → 12 languages, all on-device. |
| **Crew Shift Handover** | Show-runners between shifts | Voice memo → structured 5-section brief in seconds, no transcription service. |
| **Content Authenticity Detector** | Brand & security | Local deepfake / scam-ticket detection — zero upload. |

Plus an **NPU Dashboard** showing tokens, latency, cost-saved, and runtime.

## Quick start (Windows, Intel NPU)

```powershell
# 1. Install Foundry Local + the NPU model (one-time)
winget install Microsoft.FoundryLocal
foundry model run phi-4-mini-instruct-openvino-npu

# 2. Set up the Python venv
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 3. Launch
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe app.py
# → http://localhost:5006
```

The app probes Foundry's rotating port on startup and waits up to 240s for the
first NPU compile of Phi-4-mini.

## Notes

- **NPU is single-request** — concurrent calls return HTTP 500 *Infer Request
  is busy*. The UI serializes naturally.
- **Voice input** uses the Web Speech API (Edge / Chrome on Windows).
- **Max generation** is clamped to 480 tokens to stay inside the OpenVINO NPU
  cache window.
- **Default port is 5006** so it can run alongside other local demos.

## License

MIT.
