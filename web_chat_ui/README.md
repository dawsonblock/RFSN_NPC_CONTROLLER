# RFSN Web Chat UI

A production-clean local web chat interface for the RFSN NPC Orchestrator.

## Components

1.  **Backend Proxy**: Python FastAPI server (Port 3001). Proxies requests to the orchestrator, handles SSE normalization, and logs episodes.
2.  **Frontend**: React + Vite SPA (Port 5173). Modern dark-mode UI with streaming chat, NPC state control, and meta-data visualization.

## Prerequisites

- Python 3.9+
- Node.js 18+
- The **RFSN Orchestrator** must be running locally on port 8000.
  - Streaming Endpoint: `/api/dialogue/stream` (Discovered from repo analysis)

## Setup & Run

### 1. Start the Backend Proxy

```bash
cd web_chat_ui/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 3001
```

*The backend will log the discovered orchestrator route on startup.*

### 2. Start the Frontend

```bash
cd web_chat_ui/frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

Open your browser to `http://localhost:5173`.

## Usage

1.  **NPC Identity**: Set the Name and ID (defaults to Bouncer/npc_001).
2.  **NPC State**: Modify the JSON to change mood, affinity, etc. affecting the NPC's response.
3.  **Chat**: Type and press Enter to chat. Responses stream in real-time.
4.  **Meta Panel**: View the raw metadata (bandit key, action mode, player signal) for the current turn.
5.  **Logging**:
    - "Copy" button copies the episode JSONL to clipboard.
    - "DL" button downloads the current session's JSONL log.
    - Logs are stored in `web_chat_ui/backend/episodes/`.

## Manual Test (CURL)

You can test the backend proxy streaming directly with curl:

```bash
curl -N -X POST http://localhost:3001/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "npc_id": "npc_tester",
    "npc_name": "TestNPC",
    "player_text": "Hello friend!",
    "npc_state": {
      "mood": "happy",
      "affinity": 0.5,
      "relationship": "friend",
      "combat_active": false
    },
    "session_id": "manual-test-01"
  }'
```

You should see streaming SSE events (`event: meta` followed by `event: text` chunks).
