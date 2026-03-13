# AGENTS.md

This file is the primary project context to consult before making changes.

## Project purpose
- Gradio chat app that acts as an F1 2020 UDP telemetry advisor.
- Uses LangGraph for a single F1 advisor agent.

## Entry point
- `main.py` is the application entry point. Run with:
  - `python main.py`

## Runtime and dependencies
- Python app using:
  - `gradio`
  - `langgraph`
  - `langchain`
  - `dotenv`
  - `f1-2020-telemetry` (UDP telemetry parsing)
- Environment variables are loaded from `.env` via `load_dotenv()`.

## Core flow (LangGraph)
- Single node graph: `START -> f1_advisor -> END`.
- Advisor uses latest telemetry snapshot to answer practice/qualifying/race strategy questions.

## UDP telemetry
- Listens on UDP `127.0.0.1:20777`.
- Telemetry packets are parsed by `TelemetryParser` using `f1_2020_telemetry.packets.unpack_udp_packet`.
- Latest snapshot is merged in `TelemetryStore` so partial packets (session/lap/status) combine into a full view.

## UI
- Gradio `ChatInterface` in `main.py`.
- Title: `F1 UDP Advisor`
- Description: `Ask for practice, qualifying, or race strategy using live F1 2020 telemetry.`

## Conventions and cautions
- Do not remove UDP listener or snapshot storage logic.
- Keep advisor responses aligned with session type and telemetry freshness.
- If telemetry is missing or stale, advisor should ask for context.

## Testing
- No automated tests in this repo.
- Manual run: `python main.py` and verify:
  - UDP listener binds to `127.0.0.1:20777`
  - Chat answers reference telemetry context when available.
