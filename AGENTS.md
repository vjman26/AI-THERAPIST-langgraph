# AGENTS.md

This file is the primary project context to consult before making changes.

## Project purpose
- Full-stack F1 2020 UDP telemetry advisor.
- `server/` ingests UDP telemetry, serves WebSocket updates, and exposes an LLM chat endpoint.
- `client/` renders a live engineer dashboard with charts and an LLM chat drawer.

## Entry points
- Server: `server/main.py` (FastAPI)
- Client: `client/` (Vite + React)

## Run
- Server: `uvicorn server.main:app --reload --host 0.0.0.0 --port 8000`
- Client: `npm install` then `npm run dev` inside `client/`

## Server behavior
- UDP listener binds to `UDP_HOST`/`UDP_PORT` (defaults to `0.0.0.0:20777`).
- Parses packets with `f1_2020_telemetry.packets.unpack_udp_packet`.
- Merges partial packet data into a single telemetry snapshot.
- WebSocket: `/ws/telemetry` publishes JSON payloads.
- REST: `/api/chat` for LLM responses.

## Client behavior
- Connects to `ws://localhost:8000/ws/telemetry` by default.
- Uses `VITE_API_BASE` to override API base URL.
- Shows status cards, line charts, and a chat drawer that overlays without hiding graphs.

## Key data exposed
- Session, lap, position, tyre compound/wear, fuel, gaps, ERS, weather, safety car, damage.
- Gaps are estimated from total distance and speed.

## Conventions and cautions
- Keep telemetry field meanings aligned with the F1 2020 UDP spec.
- Chat should always remain visible without removing the dashboard context.
