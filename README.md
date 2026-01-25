<div align="center">

# 🎮 RFSN NPC Controller

<img src="https://img.shields.io/badge/version-1.4-blueviolet?style=for-the-badge" alt="Version 1.4"/>

**Production-Ready Streaming AI System for Real-Time NPC Dialogue**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-297%20Passing-success?style=flat-square&logo=pytest)](Python/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-black?style=flat-square)](https://ollama.ai/)

*Intelligent NPCs with semantic action selection, world model prediction, adaptive learning, and real-time TTS*

---

[**Features**](#-features) • [**Quick Start**](#-quick-start) • [**Architecture**](#-architecture) • [**API**](#-api-reference) • [**Learning**](#-learning-layer) • [**Performance**](#-performance)

</div>

---

## ✨ Features

### 🧠 Core Intelligence

| Feature | Description |
|---------|-------------|
| **Semantic Action Selection** | World model predicts outcomes and scores 32 discrete NPC actions |
| **Contextual Bandits (v1.4)** | LinUCB with **Sherman-Morrison optimization** (O(d²) updates) |
| **Hierarchical State Abstraction** | Maps raw state to compact categorical keys for faster learning |
| **N-Step Action Traces** | Temporal credit assignment for delayed rewards |
| **Two-Stage Policy** | Separately learns *what* to do (action) and *how* to say it (phrasing) |
| **Explicit Reward Channel** | External feedback injection (quest completions, UI buttons) |
| **Regression Guard** | Auto-freezes learning if correction/block rates exceed safety thresholds |
| **Ground-Truth Anchors** | Deterministic rewards for explicit feedback ("That's wrong", "Thank you") |

### 🎙️ Voice & Speech

| Feature | Description |
|---------|-------------|
| **Dual TTS Router** | Chatterbox-Turbo + Chatterbox-Full with automatic intensity-based selection |
| **Lazy Model Loading** | TTS/LLM engines loaded on first use (faster startup) |
| **LRU Audio Cache** | 100 clips, 5-min TTL for repeated lines |
| **Kokoro Fallback** | Graceful CPU-only degradation when CUDA unavailable |

### 🛡️ Production Hardening

| Feature | Description |
|---------|-------------|
| **Tests** | 297+ tests covering streaming, learning, safety, and persistence |
| **Safety** | Hard overrides prevent learned stupidity in combat/trust/quest contexts |
| **Debug Logging** | Structured `[BANDIT SELECT]` and `[BANDIT UPDATE]` logs for decision tracing |
| **Modular Architecture** | Routers, Services, and Learning modules cleanly separated |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (required for Kokoro TTS)
- **4GB RAM** minimum
- **Ollama** for local LLM inference

### Installation

```bash
# Clone the repository
git clone https://github.com/dawsonblock/RFSN_NPC_CONTROLLER.git
cd RFSN_NPC_CONTROLLER

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r Python/requirements-core.txt
pip install kokoro-onnx

# Install & start Ollama (macOS)
brew install ollama
ollama serve &
ollama pull llama3.2

# Launch the server
cd Python
python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8000
```

### Access Points

| Endpoint | URL |
|----------|-----|
| **API Server** | `http://localhost:8000` |
| **Dashboard** | `http://localhost:8000` |
| **Mobile UI** | `http://localhost:8080` (run `python mobile_chat/server.py`) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RFSN NPC Controller v1.4                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   FastAPI    │───▶│  Streaming   │───▶│   Voice Router       │  │
│  │   Routers    │    │   Engine     │    │  (Turbo/Full/Kokoro) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Dialogue   │    │ DequeSpeech  │    │    Audio Player      │  │
│  │   Service    │    │    Queue     │    │    (Async/Stream)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                                      │
│         ▼                    ▼                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │    State     │    │ World Model  │    │   Emotional State    │  │
│  │  Abstractor  │───▶│ (Prediction) │◀───│   (VAD + Decay)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Contextual  │    │   Action     │    │   Metrics Guard      │  │
│  │    Bandit    │───▶│   Tracer     │◀───│  (Regression Guard)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Orchestrator** | FastAPI server, lifecycle hooks | `Python/orchestrator.py` |
| **Dialogue Service** | Core dialogue pipeline | `Python/services/dialogue_service.py` |
| **State Abstractor** | Hierarchical state compression | `Python/learning/state_abstraction.py` |
| **Action Tracer** | N-step temporal credit | `Python/learning/action_trace.py` |
| **Reward Channel** | Explicit reward injection | `Python/learning/reward_signal.py` |
| **Contextual Bandit** | LinUCB (Sherman-Morrison) | `Python/learning/contextual_bandit.py` |
| **Mode Bandit** | Phrasing style learner | `Python/learning/mode_bandit.py` |
| **World Model** | Predicts state transitions | `Python/world_model.py` |

---

## 📡 API Reference

### Streaming Dialogue

```http
POST /api/dialogue/stream
Content-Type: application/json

{
  "npc_name": "Jarl Balgruuf",
  "user_input": "Tell me about Whiterun."
}
```

**Response**: Server-Sent Events (SSE)

```json
data: {"sentence": "Whiterun is a great city.", "is_final": false, "latency_ms": 150}
data: {"sentence": "We welcome all travelers.", "is_final": true, "latency_ms": 280}
```

### Memory Management

```http
GET  /api/memory/{npc_name}/stats       # Get memory statistics
POST /api/memory/{npc_name}/safe_reset  # Reset with backup
GET  /api/memory/{npc_name}/backups     # List available backups
```

---

## 🤖 Learning Layer (v1.4)

### Contextual Bandit (LinUCB)

Learns optimal actions using rich feature vectors. Uses **Sherman-Morrison formula** for O(d²) updates instead of O(d³) matrix inversion.

### Hierarchical State Abstraction

Raw state (affinity, mood, turn count, etc.) → Compact categorical key → Feature vector.

### N-Step Action Traces

Rolling buffer of (state, action, reward) tuples enables delayed credit assignment.

### Two-Stage Policy

1. **Action Bandit**: Selects *WHAT* to do (e.g., `GREET`, `OFFER`).
2. **Mode Bandit**: Selects *HOW* to say it.

| Mode | Description |
|------|-------------|
| `TERSE_DIRECT` | Short, factual responses (3-4 sentences) |
| `WARM_FRIENDLY` | Empathetic, relational responses |
| `LORE_RICH` | Detailed world-building responses |
| `PLAYFUL_WITTY` | Humorous, light-hearted responses |
| `NEUTRAL_BALANCED` | Default balanced approach |

---

## 🧪 Testing & Debugging

```bash
# Run all tests
cd Python && python -m pytest tests/ -v

# Enable debug logging
LOG_LEVEL=DEBUG python -m uvicorn orchestrator:app --port 8000
```

Debug logs include:

- `[BANDIT SELECT]` — Context ID, top-3 candidates, chosen action
- `[BANDIT UPDATE]` — Context ID, action, reward, arm update count
- `[DECISION]` — Structured decision traces in DialogueService

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details

---

<div align="center">

## 🔗 Links

[**GitHub**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER) • [**Issues**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER/issues) • [**Discussions**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER/discussions)

---

**Made with ❤️ for immersive NPC interactions**

⭐ **Star this repo if you find it useful!**

</div>
