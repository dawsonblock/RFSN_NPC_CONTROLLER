# CGW/SSL Guard — Step 1 (Forced Override) Build

Implements a deterministic Step 1 harness:

- **ThalamusGate** with structural forced bypass (forced queue checked before competition)
- **CGWRuntime** with **single-slot attended content** + atomic swap
- **Runtime.tick()** as the cycle driver (no sleep-based tests)
- **ForcedAttentionOverride** with:
  - forced-queue discipline (empty before injection)
  - precise failure modes (selection/commit/wrong commit/overtaken/contaminated/timeout)
  - event ordering validation (selection == commit cycle, slot_id matches)
- **SerialityMonitor** (multiple commits per cycle is a hard failure)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```
