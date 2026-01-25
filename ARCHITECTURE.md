# RFSN Controller Architecture

## Philosophy

The RFSN (Recursive Finite State Narrative) Controller is designed for **deterministic, bounded, and safe** NPC behavior in games. Unlike end-to-end neural approaches, we do not let LLMs make game-logic decisions directly.

## Core Pillars

1. **Symbolic State Authority**: The game world is the source of truth. The "World Model" maintains a structured representation of state (relationship, mood, threat level) that is crisp and enumerable.
2. **Bandit-Based Control**: Action selection (e.g., `ATTACK`, `GREET`, `TRADE`) is handled by Contextual Bandits (LinUCB). This provides online learning that is mathematically bounded and explainable.
3. **LLM as Actuator**: The LLM is used *only* for generation (dialogue, flavor text) downstream of the decision. It is a "renderer" of the selected action, not the decision-maker.
4. **Safety First**: A dedicated `IntentGate` and `metric_guard` ensure that neither the heuristic model nor the LLM can violate safety constraints (e.g., toxic content, breaking character).

## Data Flow

1. **Input**: User text + Game State (JSON).
2. **Abstraction**: `StateAbstractor` maps raw state to a hierarchical key (e.g., `HOSTILE|TAVERN`).
3. **Decision**: `ContextualBandit` selects the optimal `NPCAction` based on history for this abstract state.
4. **Generation**: `DialogueService` constructs a prompt forcing the LLM to execute the chosen action.
5. **Output**: Streamed text + `FINAL_JSON` verification block.
6. **Learning**: Explicit rewards (`RewardChannel`) and implicit outcomes update the bandit policy.

## Why No Black Boxes?

- **Debuggability**: If an NPC acts weirdly, we can check the exact bandit weights for that state key.
- **Safety**: Semantic limits are hard-coded in the `IntentGate`.
- **Performance**: Bandits are orders of magnitude faster than LLM reasoning loops.
