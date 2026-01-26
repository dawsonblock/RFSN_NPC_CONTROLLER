<div align="center">

# 🎮 RFSN NPC Controller

<img src="https://img.shields.io/badge/version-2.0-blueviolet?style=for-the-badge" alt="Version 2.0"/>

**Production-Ready Streaming AI System for Real-Time NPC Dialogue**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-314%20Passing-success?style=flat-square&logo=pytest)](Python/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-black?style=flat-square)](https://ollama.ai/)

*Intelligent NPCs with symbolic state authority, single-policy learning, traceable decisions, and real-time TTS*

---

[**Features**](#-features) • [**Quick Start**](#-quick-start) • [**Architecture**](#-architecture) • [**Learning**](#-learning-system-v20) • [**API**](#-api-reference) • [**Testing**](#-testing)

</div>

---

## ✨ Features

### 🧠 Learning System (v2.0)

| Feature | Description |
|---------|-------------|
| **Single PolicyOwner** | One authoritative policy for action selection—no split-brain |
| **DecisionRecord** | Every turn emits immutable record with unique `decision_id` |
| **Reward Binding** | Rewards require `decision_id`; unknown IDs are quarantined |
| **N-Step Credit** | `ActionTracer` propagates rewards backward via decision_id |
| **Versioned Abstraction** | `StateAbstractor` v2.0 with stable schema + bucket rules |
| **Offline Evaluation** | `evaluate_logs.py` computes metrics from JSONL logs |
| **Deterministic Mode** | `LEARNING_DISABLED=1` for reproducible testing |

### 🎯 Decision & Control

| Feature | Description |
|---------|-------------|
| **Symbolic State Authority** | Game world is source of truth, not the LLM |
| **LinUCB Bandits** | Sherman-Morrison O(d²) updates with UCB exploration |
| **World Model Prediction** | Scores 32 discrete NPCActions per state |
| **Safety Overrides** | Hard limits in combat/trust/quest contexts |
| **MetricsGuard** | Auto-freezes learning if regression detected |

### 🎙️ Voice & Speech

| Feature | Description |
|---------|-------------|
| **Dual TTS Router** | Chatterbox-Turbo + Chatterbox-Full with intensity selection |
| **Lazy Loading** | TTS/LLM engines loaded on first use |
| **LRU Audio Cache** | 100 clips, 5-min TTL |
| **Kokoro Fallback** | CPU-only degradation when CUDA unavailable |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **4GB RAM** minimum
- **Ollama** for local LLM inference

### Installation

```bash
# Clone
git clone https://github.com/dawsonblock/RFSN_NPC_CONTROLLER.git
cd RFSN_NPC_CONTROLLER

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r Python/requirements-core.txt
pip install kokoro-onnx

# Ollama (macOS)
brew install ollama && ollama serve &
ollama pull llama3.2

# Launch
cd Python && python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | URL |
|----------|-----|
| **API Server** | `http://localhost:8000` |
| **Dashboard** | `http://localhost:8000` |
| **Mobile UI** | `http://localhost:8080` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RFSN NPC Controller v2.0                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input → StateAbstractor → PolicyOwner.select_action() → LLM Render   │
│                                    │                                    │
│                             DecisionRecord                              │
│                                    │                                    │
│                         ┌──────────▼──────────┐                         │
│                         │   decisions.jsonl   │                         │
│                         └──────────┬──────────┘                         │
│                                    │                                    │
│   Reward → RewardChannel (validates decision_id) → PolicyOwner.update() │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Path |
|-----------|---------|------|
| **PolicyOwner** | Single authoritative policy | `learning/policy_owner.py` |
| **DecisionRecord** | Immutable decision with decision_id | `learning/decision_record.py` |
| **ActionTracer** | N-step credit via decision_id | `learning/action_trace.py` |
| **RewardChannel** | Validated reward ingestion | `learning/reward_signal.py` |
| **StateAbstractor** | Versioned state compression | `learning/state_abstraction.py` |
| **ContextualBandit** | LinUCB arm learner | `learning/contextual_bandit.py` |
| **WorldModel** | State transition prediction | `world_model.py` |
| **DecisionIndex** | Fast log lookup | `tools/decision_index.py` |
| **OfflineEvaluator** | Compute metrics from logs | `tools/evaluate_logs.py` |

---

## 📚 Learning System (v2.0)

### Data Flow

1. **State** → `StateAbstractor` creates versioned abstract key
2. **Selection** → `PolicyOwner.select_action()` produces `DecisionRecord`
3. **Persistence** → Record logged to `decisions.jsonl` with `decision_id`
4. **Execution** → LLM renders the chosen action
5. **Reward** → `RewardChannel.submit_reward(decision_id, reward)`
6. **Update** → `PolicyOwner.update()` using validated decision_id

### Log Artifacts

| File | Contents |
|------|----------|
| `data/learning/decisions.jsonl` | Every DecisionRecord |
| `data/learning/rewards.jsonl` | Validated rewards |
| `data/learning/rewards_quarantine.jsonl` | Rejected rewards |

### Environment Flags

```bash
LEARNING_DISABLED=1   # Deterministic mode (no exploration)
STRICT_LEARNING=1     # Reject updates without explicit rewards
STRICT_REWARDS=1      # Reject rewards with unknown decision_id
```

### Offline Evaluation

```bash
python -m tools.evaluate_logs --logdir data/learning --out report.json
```

Outputs: action distribution, reward stats, exploration ratio, safety veto rate, regret proxy.

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
data: {"sentence": "Whiterun is a great city.", "is_final": false}
data: {"sentence": "We welcome all travelers.", "is_final": true}
```

### Memory Management

```http
GET  /api/memory/{npc_name}/stats
POST /api/memory/{npc_name}/safe_reset
GET  /api/memory/{npc_name}/backups
```

---

## 🧪 Testing

```bash
# Run all tests (314+)
cd Python && python -m pytest tests/ -v

# Run PolicyOwner tests only (17 tests)
python -m pytest tests/test_policy_owner.py -v

# Enable debug logging
LOG_LEVEL=DEBUG python -m uvicorn orchestrator:app --port 8000
```

### Debug Logs

- `[POLICY SELECT]` — decision_id, action, exploration flag
- `[POLICY UPDATE]` — decision_id, reward, scaled reward
- `[REWARD]` — decision_id, reward, source

---

## 📁 Project Structure

```
Python/
├── learning/
│   ├── policy_owner.py       # Single policy entrypoint
│   ├── decision_record.py    # DecisionRecord + logger
│   ├── action_trace.py       # N-step credit assignment
│   ├── reward_signal.py      # Validated reward channel
│   ├── state_abstraction.py  # Versioned state bucketing
│   └── contextual_bandit.py  # LinUCB implementation
├── tools/
│   ├── decision_index.py     # Fast log lookup
│   └── evaluate_logs.py      # Offline evaluation
├── config/
│   └── state_abstraction.yaml
├── services/
│   └── dialogue_service.py   # Main dialogue pipeline
├── tests/
│   └── test_policy_owner.py  # 17 tests for decision_id flow
└── orchestrator.py           # FastAPI server
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details

---

<div align="center">

[**GitHub**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER) • [**Issues**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER/issues) • [**Discussions**](https://github.com/dawsonblock/RFSN_NPC_CONTROLLER/discussions)

---

**Made with ❤️ for immersive NPC interactions**

⭐ **Star this repo if you find it useful!**

</div>
