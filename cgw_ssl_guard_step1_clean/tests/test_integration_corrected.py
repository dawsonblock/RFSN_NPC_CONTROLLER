from cgw_ssl_guard import (
    SimpleEventBus, ThalamusGate, CGWRuntime, Runtime,
    Candidate, SelfModel, ForcedAttentionOverride, SerialityMonitor
)

def test_forced_override_integrated():
    event_bus = SimpleEventBus()
    gate = ThalamusGate(event_bus)
    cgw = CGWRuntime(event_bus)
    runtime = Runtime(gate, cgw, event_bus, lambda: SelfModel(goals=["test"]))

    seriality = SerialityMonitor()
    event_bus.on("CGW_COMMIT", seriality.on_commit)

    gate.submit_candidate(Candidate(
        slot_id="normal_001",
        source_module="PLANNING",
        content_payload=b"NORMAL_CONTENT",
        saliency=0.7
    ))

    assert runtime.tick() is True
    assert cgw.get_current_state().content_id() == "normal_001"

    override = ForcedAttentionOverride(runtime, event_bus)
    result = override.execute(max_wait_cycles=5)

    assert result.success, f"Override failed: {result}"
    assert result.latency_cycles <= 3
    assert not result.selection_failed
    assert not result.commit_failed

    assert result.selection_event is not None
    assert result.commit_event is not None
    assert result.selection_event.cycle_id == result.commit_event["cycle_id"]
    assert result.selection_event.slot_id == result.commit_event["slot_id"]

    final = cgw.get_current_state()
    assert final.content_id() == result.probe_slot_id
    assert final.causal_trace.forced_override is True

    runtime.run_cycles(10)
    for cid, count in seriality.commits_per_cycle.items():
        assert count == 1, f"Cycle {cid} had {count} commits"
