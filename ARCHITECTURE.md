# RFSN Controller Architecture

## Philosophy

The RFSN (Recursive Finite State Narrative) Controller is designed for **deterministic, bounded, and safe** NPC behavior in games. Unlike end-to-end neural approaches, we do not let LLMs make game-logic decisions directly.

## Core Pillars

1. **Symbolic State Authority**: The game world is the source of truth. The "World Model" maintains a structured representation of state (relationship, mood, threat level) that is crisp and enumerable.
2. **Single Policy Owner**: Action selection is handled by a single `PolicyOwner` module using LinUCB contextual bandits. This eliminates policy split-brain and provides online learning that is mathematically bounded and explainable.
3. **LLM as Actuator**: The LLM is used *only* for generation (dialogue, flavor text) downstream of the decision. It is a "renderer" of the selected action, not the decision-maker.
4. **Safety First**: A dedicated `IntentGate` and `MetricsGuard` ensure that neither the heuristic model nor the LLM can violate safety constraints (e.g., toxic content, breaking character).

## Data Flow

```
Input → StateAbstractor → PolicyOwner.select_action() → DecisionRecord → LLM Renderer → Output
                                    ↓
                              decisions.jsonl
                                    ↓
           Reward → RewardChannel (validated by decision_id) → PolicyOwner.update()
```

1. **Input**: User text + Game State (JSON).
2. **Abstraction**: `StateAbstractor` maps raw state to a versioned hierarchical key (e.g., `LOCATION:tavern|QUEST_STATE:active`).
3. **Decision**: `PolicyOwner.select_action()` selects the optimal `NPCAction` and creates a `DecisionRecord` with unique `decision_id`.
4. **Generation**: `DialogueService` constructs a prompt forcing the LLM to execute the chosen action.
5. **Output**: Streamed text + `FINAL_JSON` verification block.
6. **Learning**: Rewards submitted via `RewardChannel` must include `decision_id` for proper attribution. Unknown decision_ids are quarantined.

## Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| PolicyOwner | `learning/policy_owner.py` | **SINGLE** authoritative policy for selection + update |
| DecisionRecord | `learning/decision_record.py` | Join key for reward attribution and trace credit |
| ActionTracer | `learning/action_trace.py` | N-step credit assignment using decision_ids |
| RewardChannel | `learning/reward_signal.py` | Validated reward ingestion (requires decision_id) |
| StateAbstractor | `learning/state_abstraction.py` | Versioned, deterministic state abstraction |
| DecisionIndex | `tools/decision_index.py` | Fast lookup from JSONL logs |
| OfflineEvaluator | `tools/evaluate_logs.py` | Computes learning metrics from logs |

## Logging Artifacts

| File | Contents |
|------|----------|
| `data/learning/decisions.jsonl` | Every turn's DecisionRecord with decision_id |
| `data/learning/rewards.jsonl` | Validated rewards bound to decision_ids |
| `data/learning/rewards_quarantine.jsonl` | Rejected rewards with unknown decision_ids |

## Environment Flags

- `STRICT_LEARNING=1`: Reject updates without explicit rewards
- `LEARNING_DISABLED=1`: Deterministic mode for testing (no bandit exploration)
- `STRICT_REWARDS=1`: Reject rewards with unknown decision_ids

## Why No Black Boxes?

- **Debuggability**: If an NPC acts weirdly, we can trace the exact decision_id and replay it.
- **Safety**: Semantic limits are hard-coded in the `IntentGate`.
- **Performance**: Bandits are orders of magnitude faster than LLM reasoning loops.
- **Reproducibility**: With `LEARNING_DISABLED=1`, repeated runs produce identical decisions.
