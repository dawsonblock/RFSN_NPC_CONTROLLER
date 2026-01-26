from cgw_ssl_guard import SimpleEventBus, ThalamusGate, CGWRuntime, Runtime, Candidate, SelfModel, SerialityMonitor
from cgw_ssl_guard.forced_override import EventCollector

def test_forced_queue_discipline():
    event_bus = SimpleEventBus()
    gate = ThalamusGate(event_bus)
    assert len(gate.forced_queue) == 0
    gate.inject_forced_signal("TEST", b"test")
    assert len(gate.forced_queue) == 1

def test_event_ordering():
    event_bus = SimpleEventBus()
    gate = ThalamusGate(event_bus)
    cgw = CGWRuntime(event_bus)
    runtime = Runtime(gate, cgw, event_bus, lambda: SelfModel(goals=["x"]))

    selection = {"t": None, "cid": None, "sid": None}
    commit = {"t": None, "cid": None, "sid": None}

    def on_sel(e):
        selection["t"] = e.timestamp
        selection["cid"] = e.cycle_id
        selection["sid"] = e.slot_id

    def on_com(e):
        commit["t"] = e["timestamp"]
        commit["cid"] = e["cycle_id"]
        commit["sid"] = e["slot_id"]

    event_bus.on("GATE_SELECTION", on_sel)
    event_bus.on("CGW_COMMIT", on_com)

    gate.submit_candidate(Candidate(slot_id="cand_1", source_module="TEST", content_payload=b"abc", saliency=0.5))
    assert runtime.tick() is True

    assert selection["t"] is not None
    assert commit["t"] is not None
    assert selection["t"] <= commit["t"]
    assert selection["cid"] == commit["cid"]
    assert selection["sid"] == commit["sid"]

def test_seriality_monitor_coverage():
    event_bus = SimpleEventBus()
    gate = ThalamusGate(event_bus)
    cgw = CGWRuntime(event_bus)
    runtime = Runtime(gate, cgw, event_bus, lambda: SelfModel(goals=["x"]))

    seriality = SerialityMonitor()
    event_bus.on("CGW_COMMIT", seriality.on_commit)

    gate.submit_candidate(Candidate(slot_id="cand_1", source_module="TEST", content_payload=b"abc", saliency=0.5))
    runtime.run_cycles(10)

    assert len(seriality.commits_per_cycle) > 0
    for cid, count in seriality.commits_per_cycle.items():
        assert count == 1

def test_winner_is_forced_flag():
    event_bus = SimpleEventBus()
    gate = ThalamusGate(event_bus)

    collector = EventCollector()
    event_bus.on("GATE_SELECTION", collector.on_gate_selection)

    forced_id = gate.inject_forced_signal("TEST", b"forced")
    winner, reason = gate.select_winner()
    ev = collector.find_selection(forced_id)
    assert ev is not None
    assert ev.winner_is_forced is True

    gate.submit_candidate(Candidate(slot_id="cand_1", source_module="TEST", content_payload=b"abc", saliency=0.5))
    winner, reason = gate.select_winner()
    last = collector.selections[-1]
    assert last.winner_is_forced is False
