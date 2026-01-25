<div align="center">

# 🎮 RFSN NPC Controller

<img src="https://img.shields.io/badge/version-10.2-blueviolet?style=for-the-badge" alt="Version 10.2"/>

**Production-Ready Streaming AI System for Real-Time NPC Dialogue**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-280%20Passing-success?style=flat-square&logo=pytest)](Python/tests/)
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
| **Contextual Bandits** | Thompson sampling with adaptive exploration learns optimal dialogue styles |
| **Temporal Memory** | Short-term experience buffer enables anticipatory reasoning |
| **Hybrid NLU** | LLM-powered intent classification with regex fallback via Ollama |
| **Emotional Modeling** | VAD-based (Valence/Arousal/Dominance) emotional state with decay |
| **Sentiment Tracking** | Multi-player longitudinal sentiment analysis with trend detection |

### 🎙️ Voice & Speech

| Feature | Description |
|---------|-------------|
| **Dual TTS Router** | Chatterbox-Turbo + Chatterbox-Full with automatic intensity-based selection |
| **Lazy Model Loading** | Full model (~2GB VRAM) loaded only on first HIGH intensity request |
| **LRU Audio Cache** | 100 clips, 5-min TTL for repeated lines |
| **Kokoro Fallback** | Graceful CPU-only degradation when CUDA unavailable |

### 🛡️ Production Hardening

- ✅ **280+ Tests** — Comprehensive coverage including streaming, learning, world model, and persistence
- ✅ **Dot-Path Config** — Nested config access (`llm.temperature`) with hot-reload support
- ✅ **Zero Race Conditions** — Deque+Condition queue pattern eliminates task_done/join bugs
- ✅ **Atomic State Swaps** — RuntimeState prevents half-applied config during hot reloads
- ✅ **Full Persistence** — Temporal memory, emotional states, and bandit weights survive restarts
- ✅ **Safety Rules** — Hard overrides prevent learned stupidity in combat/trust/quest contexts

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
│                     RFSN NPC Controller v10.2                        │
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
│  │   Bandit     │    │Action Scorer │    │  Sentiment Tracker   │  │
│  │   Learner    │───▶│ (32 Actions) │◀───│  (Multi-Player)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Orchestrator** | FastAPI server, lifecycle hooks | `Python/orchestrator.py` |
| **Streaming Engine** | Token processing, sentence detection | `Python/streaming_engine.py` |
| **Voice Router** | Dual-TTS with lazy load, LRU cache | `Python/voice_router.py` |
| **World Model** | Predicts state transitions | `Python/world_model.py` |
| **Action Scorer** | Scores 32 candidate actions | `Python/action_scorer.py` |
| **NPC Action Bandit** | Thompson sampling learner | `Python/learning/npc_action_bandit.py` |
| **Temporal Memory** | Short-term experience buffer | `Python/learning/temporal_memory.py` |
| **Emotional State** | VAD modeling with decay | `Python/emotional_tone.py` |
| **Sentiment Tracker** | Longitudinal player analysis | `Python/learning/sentiment_tracker.py` |
| **Intent Extraction** | Hybrid LLM+regex classification | `Python/intent_extraction.py` |
| **State Machine** | Invariant-validated state transitions | `Python/state_machine.py` |
| **Hot Config** | Dot-path nested config with hot-reload | `Python/hot_config.py` |

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

```
data: {"sentence": "Whiterun is a great city.", "is_final": false, "latency_ms": 150}
data: {"sentence": "We welcome all travelers.", "is_final": true, "latency_ms": 280}
```

### Memory Management

```http
GET  /api/memory/{npc_name}/stats       # Get memory statistics
POST /api/memory/{npc_name}/safe_reset  # Reset with backup
GET  /api/memory/{npc_name}/backups     # List available backups
```

### Performance Tuning

```http
POST /api/tune-performance
{
  "temperature": 0.7,
  "max_tokens": 150,
  "max_queue_size": 3
}
```

### Health & Metrics

```http
GET /api/health           # Health check
WS  /ws/metrics           # WebSocket metrics stream
```

---

## 🤖 Learning Layer

### Contextual Bandit

The system uses Thompson sampling with adaptive exploration to learn optimal dialogue styles per NPC:

| Mode | Description |
|------|-------------|
| `TERSE_DIRECT` | Short, factual responses (3-4 sentences) |
| `WARM_FRIENDLY` | Empathetic, relational responses |
| `LORE_RICH` | Detailed world-building responses |
| `PLAYFUL_WITTY` | Humorous, light-hearted responses |
| `FORMAL_RESPECTFUL` | Distant, proper responses |
| `NEUTRAL_BALANCED` | Default balanced approach |

### Safety Rules

Hard overrides prevent learned stupidity:

| Condition | Override |
|-----------|----------|
| **Combat + Fear > 0.7** | Forces `FLEE` action |
| **Trust < 0.1** | Forbids `ACCEPT`, `OFFER`, `HELP` |
| **Quest Active** | Biases toward `HELP`, `AGREE` |

---

## 🎙️ Voice Router

Intelligent dual-TTS engine with automatic model selection:

| Intensity | Engine | Use Case | Exaggeration |
|-----------|--------|----------|--------------|
| **LOW** | Chatterbox-Turbo | Guards, shopkeepers, barks | 0.3 |
| **MEDIUM** | Chatterbox-Turbo | Companion casual chat | 0.6 |
| **HIGH** | Chatterbox-Full | Memory callbacks, relationship moments | 0.8 |

**Optimizations:**

- 🚀 **Lazy Loading** — Full model loaded only when needed
- 💾 **LRU Cache** — 100 clips with 5-minute TTL
- ⚡ **Precompute** — Intensity cached for 5 seconds
- 🔄 **Fallback** — Graceful Kokoro degradation

---

## ⚡ Performance

### Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| First Token Latency | <1.5s | ~1.2s |
| Sentence Detection | <50ms | ~30ms |
| TTS Processing | <100ms | ~80ms |
| Queue Throughput | 10/s | 12/s |

### Optimizations

- **Deque+Condition Queue** — Eliminates race conditions
- **Atomic Drop Policy** — Drop runs under same lock as worker
- **Pre-compiled Regex** — No hot-path compilation overhead
- **Config Snapshots** — Per-request snapshots prevent mid-stream changes

---

## 🧪 Testing

```bash
# Run all tests
cd Python && python -m pytest tests/ -v

# Specific categories
pytest tests/test_learning*.py -v          # Learning layer
pytest tests/test_world_model*.py -v       # World model
pytest tests/test_voice_router.py -v       # Voice routing
pytest tests/test_production.py -v         # Production scenarios
```

### Coverage

| Category | Tests |
|----------|-------|
| Core Functionality | 165+ |
| Learning Layer | 45+ |
| World Model | 25+ |
| Voice Router | 30+ |
| State Machine & Config | 15 |
| **Total** | **280+** |

---

## 🔧 Configuration

### `config.json`

```json
{
  "llm": {
    "backend": "ollama",
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "temperature": 0.7,
    "max_tokens": 150
  },
  "tts": {
    "backend": "chatterbox",
    "chatterbox": {
      "device": "cuda",
      "default_exaggeration": 0.5
    }
  },
  "learning": {
    "temporal_memory": { "enabled": true, "max_size": 50 },
    "nuance_variants": { "enabled": true }
  }
}
```

**Dot-path access** — Access nested values with `config.get("llm.temperature")

```

### Environment Variables

```bash
export RFSN_PORT=8000
export RFSN_HOST=0.0.0.0
export RFSN_LOG_LEVEL=DEBUG
```

---

## 📁 Project Structure

```
RFSN_NPC_CONTROLLER/
├── Python/
│   ├── orchestrator.py         # FastAPI server
│   ├── streaming_engine.py     # Core streaming logic
│   ├── voice_router.py         # Dual-TTS routing
│   ├── world_model.py          # State prediction
│   ├── action_scorer.py        # Action evaluation
│   ├── emotional_tone.py       # VAD emotional state
│   ├── intent_extraction.py    # Hybrid NLU
│   ├── learning/               # Contextual bandit layer
│   │   ├── npc_action_bandit.py
│   │   ├── temporal_memory.py
│   │   ├── sentiment_tracker.py
│   │   └── event_logger.py
│   └── tests/                  # 290+ tests
├── Dashboard/                  # Metrics visualization
├── mobile_chat/                # iOS-optimized UI
├── config.json                 # Configuration
└── README.md
```

---

## 📈 Changelog

### v10.2 (Current) — Surgical Upgrade & Stabilization

- **NPCAction Case Fix** — State machine now normalizes action case correctly
- **Dot-Path Config** — Nested config access (`llm.temperature`) with hot-reload
- **Prompt Consolidation** — Removed duplicate `prompting/` module (–692 LOC)
- **IntentGate Optimization** — Per-sentence validation instead of per-chunk
- **Reward Normalization** — Per-component logging with bounded output
- 280+ tests with new state machine and config coverage

### v10.1 — Voice Router & Optimizations

- **Dual-TTS Voice Router** with lazy loading and LRU cache
- **Intensity-based routing** between Turbo and Full models
- **Precomputation caching** for stable NPC states

### v10.0 — Persistence & Emotional States

- **Temporal Memory Persistence** across restarts
- **VAD Emotional State** with time-based decay
- **LLM Intent Classification** via Ollama
- **Sentiment Tracking** with trend detection
- **Adaptive Exploration** decay from 30% to 2%

### v9.0 — Thread-Safe Queue Rewrite

- **Deque+Condition queue** replaces queue.Queue
- **Atomic drop policy** eliminates race conditions
- **RuntimeState** for safe config hot-reloads

[View Full Changelog](CHANGELOG.md)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

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
