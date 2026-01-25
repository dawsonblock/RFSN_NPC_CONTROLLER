<div align="center">

# 🎮 RFSN NPC Controller

<img src="https://img.shields.io/badge/version-1.3-blueviolet?style=for-the-badge" alt="Version 1.3"/>

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
| **Contextual Bandits (v1.3)** | LinUCB with feature vectors (mood, affinity, phase) |
| **Two-Stage Policy (v1.3)** | Separately learns *what* to do (action) and *how* to say it (phrasing) |
| **Regression Guard (v1.3)** | Auto-freezes learning if correction/block rates exceed safety thresholds |
| **Ground-Truth Anchors** | Deterministic rewards for explicit feedback ("That's wrong", "Thank you") |
| **Temporal Memory** | Short-term experience buffer enables anticipatory reasoning |
| **Hybrid NLU** | LLM-powered intent classification with regex fallback via Ollama |

### 🎙️ Voice & Speech

| Feature | Description |
|---------|-------------|
| **Dual TTS Router** | Chatterbox-Turbo + Chatterbox-Full with automatic intensity-based selection |
| **Lazy Model Loading** | Full model (~2GB VRAM) loaded only on first HIGH intensity request |
| **LRU Audio Cache** | 100 clips, 5-min TTL for repeated lines |
| **Kokoro Fallback** | Graceful CPU-only degradation when CUDA unavailable |

### 🛡️ Production Hardening

| Feature | Description |
|---------|-------------|
| **Tests** | 297+ tests covering streaming, learning, safety, and persistence |
| **Safety** | Hard overrides prevent learned stupidity in combat/trust/quest contexts |
| **Consistency** | Atomic state swaps and strict JSON schema validation |
| **Performance** | Zero race conditions via Deque+Condition pattern |

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

### Docker

```bash
docker build -t rfsn-npc .
docker run -p 8000:8000 rfsn-npc
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RFSN NPC Controller v1.3                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   FastAPI    │───▶│  Streaming   │───▶│   Voice Router       │  │
│  │   Server     │    │   Engine     │    │  (Turbo/Full/Kokoro) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Hybrid     │    │ DequeSpeech  │    │    Audio Player      │  │
│  │   NLU Gate   │    │    Queue     │    │    (Async/Stream)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                                      │
│         ▼                    ▼                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Temporal   │    │ World Model  │    │   Emotional State    │  │
│  │   Memory     │───▶│ (Prediction) │◀───│   (VAD + Decay)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Contextual  │    │   Two-Stage  │    │   Metrics Guard      │  │
│  │    Bandit    │───▶│    Policy    │◀───│  (Regression Guard)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Orchestrator** | FastAPI server, lifecycle hooks | `Python/orchestrator.py` |
| **Streaming Engine** | Token processing, sentence detection | `Python/streaming_engine.py` |
| **Contextual Bandit** | LinUCB learner for actions | `Python/learning/contextual_bandit.py` |
| **Mode Bandit** | Phrasing style learner | `Python/learning/mode_bandit.py` |
| **Metrics Guard** | Regression monitoring | `Python/learning/metrics_guard.py` |
| **World Model** | Predicts state transitions | `Python/world_model.py` |
| **Action Scorer** | Scores 32 candidate actions | `Python/action_scorer.py` |
| **Voice Router** | Dual-TTS with lazy load, LRU cache | `Python/voice_router.py` |

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

## 🤖 Learning Layer (v1.3)

### Contextual Bandit (LinUCB)

Learns optimal actions using rich feature vectors (Mood, Affinity, Phase, Safety). Uses **counterfactual logging** for offline analysis.

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

### Regression Guard

Protect the model from bad updates:

- **Freeze**: If User Correction Rate > 25%
- **Reduce**: If Reward drops > 15%

---

## 🧪 Testing

```bash
# Run all tests
cd Python && python -m pytest tests/ -v

# Specific categories
pytest tests/test_learning_depth.py -v     # v1.3 Learning tests
pytest tests/test_production.py -v         # Production scenarios
```

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
