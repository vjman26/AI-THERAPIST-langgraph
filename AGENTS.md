# AGENTS.md

This file is the primary project context to consult before making changes.

## Project purpose
- A Gradio chat app that routes user messages through a LangGraph workflow.
- Two primary modes: therapist (emotional support with intake form tooling) and logical (facts-only).

## Entry point
- `main.py` is the application entry point. Run with:
  - `python main.py`

## Runtime and dependencies
- Python app using:
  - `gradio`
  - `langgraph`
  - `langchain`
  - `dotenv`
  - `pydantic`
- Environment variables are loaded from `.env` via `load_dotenv()`.

## Core flow (LangGraph)
- `classify_message` uses a structured LLM output to label a message as `emotional`, `logical`, or `exit`.
- `router` maps classification to `therapist`, `logical`, or `end_conversation` nodes.
- Graph flow: `START -> classifier -> router -> (therapist|logical|end_conversation) -> END`.

## Therapist agent behavior
- Uses tool calls:
  - `fill_patient_form(form_data: PatientForm)` writes to `users.csv` and returns a unique ID.
  - `get_patient_record(unique_id: str)` reads from `users.csv` to retrieve a record.
- When user provides a unique ID, it should retrieve the record immediately.
- When user provides name and symptoms, it should fill the form and infer a condition.

## Logical agent behavior
- Provides concise, factual responses without emotional support.

## End conversation agent behavior
- Returns a warm goodbye with a gentle closing proverb.

## Data storage
- `users.csv` is the persistent store for patient intake forms.
- Records include: `name`, `symptoms`, `condition`, `date_and_time`, `unique_id`.

## UI
- Gradio `ChatInterface` in `main.py`.
- Title: `AI-Therapist`
- Description: `How can i help you today!`
- Example prompts: "I'm feeling very sad today.", "Ive already texted you", "bye"

## Conventions and cautions
- Keep tool behavior deterministic and data format stable (`users.csv` fields).
- Do not break the routing keys (`emotional`, `logical`, `exit`) or node names.
- Maintain compatibility with the current LangGraph state shape:
  - `State = { messages: list[BaseMessage], message_type: str | None }`

## Testing
- No automated tests in this repo.
- Manual run: `python main.py` and verify chatbot behavior in the Gradio UI.
