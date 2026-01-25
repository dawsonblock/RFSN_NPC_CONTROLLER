import re
import json
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from runtime_state import Runtime, RuntimeState
from world_model import StateSnapshot, NPCAction, PlayerSignal
from learning.reward_model import RewardModel
from learning.learning_contract import LearningUpdate, EvidenceType, WriteGateError
from learning.mode_bandit import PhrasingMode, ModeContext
from learning.contextual_bandit import BanditContext, ContextualBandit
from learning.metrics_guard import MetricsGuard
from learning.temporal_memory import TemporalMemory
from learning import ActionMode, RewardSignals
from reward_shaping import RewardAccumulator
from memory_manager import ConversationManager
from multi_npc import MultiNPCManager
from prometheus_metrics import inc_tokens, observe_first_token, observe_request_duration, inc_errors
from event_recorder import EventRecorder, EventType
from llm_action_prompts import render_action_block
from emotional_tone import get_emotion_manager
from intent_extraction import IntentType, SafetyFlag, IntentExtractor
from intent_extraction import IntentType, SafetyFlag, IntentExtractor
from streaming_engine import RFSNState
# New Learning Modules
from learning.state_abstraction import StateAbstractor
from learning.action_trace import ActionTracer
from learning.reward_signal import RewardChannel

logger = logging.getLogger("orchestrator")

# --- Debug Logging Helpers ---
def _log_decision(npc_name: str, step: str, details: Dict[str, Any]):
    """Structured debug log for decision tracing."""
    logger.debug(f"[DECISION] [{npc_name}] {step}: {json.dumps(details, default=str)}")

def _extract_tail_json_payload(raw_text: str) -> tuple[Optional[Dict[str, Any]], bool]:
    """
    Extract authoritative FINAL_JSON block from LLM output.
    """
    payload = None
    is_verified = False
    
    try:
        # Try strict FINAL_JSON: format first (preferred)
        final_json_match = re.search(
            r'FINAL_JSON:\s*```json\s*(\{.*?\})\s*```',
            raw_text, re.DOTALL | re.IGNORECASE
        )
        
        if final_json_match:
            json_text = final_json_match.group(1)
            payload = json.loads(json_text)
            
            # Verify required fields exist
            if all(k in payload for k in ["line", "action"]):
                is_verified = True
                # Validate confidence if present
                if "confidence" in payload:
                    conf = payload["confidence"]
                    if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
                        payload["confidence"] = 0.5  # Default if invalid
        else:
            # Fallback: try legacy ```json format (but mark as unverified)
            legacy_match = re.search(
                r'```json\s*(\{.*?\})\s*```',
                raw_text, re.DOTALL | re.IGNORECASE
            )
            if legacy_match:
                json_text = legacy_match.group(1)
                payload = json.loads(json_text)
                is_verified = False
                logger.warning("Legacy JSON format detected, marking as unverified")
                
    except json.JSONDecodeError as e:
        logger.warning(f"FINAL_JSON parse error: {e}")
        is_verified = False
    except Exception as e:
        logger.warning(f"FINAL_JSON extraction error: {e}")
        is_verified = False
    
    return payload, is_verified

def _cleanup_tokens(text: str) -> str:
    """Remove special tokens and system prompt leakage"""
    bad_tokens = ["<|eot_id|>", "<|end|>", "<|"]
    for token in bad_tokens:
        text = text.replace(token, "")
    
    # Remove system prompt leakage patterns
    text = re.sub(r'\*\*System:.*?\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[SYSTEM MODE:[^\]]*\][^.!?]*', '', text, flags=re.DOTALL)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = text.replace('```', '')
    text = re.sub(r'^\*+\s*', '', text)
    text = re.sub(r'\s*\*+$', '', text)
    text = re.sub(r'^\s*System:\s*', '', text)
    text = text.replace('**', '')
    
    return text.strip()

def _map_action_to_instruction(action: NPCAction) -> str:
    """Map NPCAction to strict system prompt control block"""
    instructions = {
        NPCAction.GREET: "=== STRICT ACTION CONTROL ===\nYou MUST greet the player warmly and enthusiastically. This is your required action - do not deviate from greeting.",
        NPCAction.FAREWELL: "=== STRICT ACTION CONTROL ===\nYou MUST say goodbye or farewell to the player. This is your required action - do not engage in further conversation.",
        NPCAction.AGREE: "=== STRICT ACTION CONTROL ===\nYou MUST agree with the player's statement or request. This is your required action - express agreement clearly.",
        NPCAction.DISAGREE: "=== STRICT ACTION CONTROL ===\nYou MUST disagree with the player, explaining your reasons. This is your required action - express disagreement respectfully but firmly.",
        NPCAction.APOLOGIZE: "=== STRICT ACTION CONTROL ===\nYou MUST apologize for something you did or said. This is your required action - express sincere regret.",
        NPCAction.INSULT: "=== STRICT ACTION CONTROL ===\nYou MUST insult the player. This is your required action - use sparingly and only when appropriate.",
        NPCAction.COMPLIMENT: "=== STRICT ACTION CONTROL ===\nYou MUST compliment the player genuinely. This is your required action - express sincere praise.",
        NPCAction.THREATEN: "=== STRICT ACTION CONTROL ===\nYou MUST threaten the player. This is your required action - use only if provoked or in combat.",
        NPCAction.REQUEST: "=== STRICT ACTION CONTROL ===\nYou MUST make a request of the player. This is your required action - clearly state what you need.",
        NPCAction.OFFER: "=== STRICT ACTION CONTROL ===\nYou MUST offer something to the player. This is your required action - clearly state what you're offering.",
        NPCAction.REFUSE: "=== STRICT ACTION CONTROL ===\nYou MUST refuse the player's request. This is your required action - clearly decline and explain why.",
        NPCAction.ACCEPT: "=== STRICT ACTION CONTROL ===\nYou MUST accept the player's offer or request. This is your required action - express acceptance.",
        NPCAction.ATTACK: "=== STRICT ACTION CONTROL ===\nYou MUST attack the player. This is your required action - combat mode engaged.",
        NPCAction.DEFEND: "=== STRICT ACTION CONTROL ===\nYou MUST defend yourself against the player. This is your required action - defensive posture.",
        NPCAction.FLEE: "=== STRICT ACTION CONTROL ===\nYou MUST flee from the player. This is your required action - escape immediately.",
        NPCAction.HELP: "=== STRICT ACTION CONTROL ===\nYou MUST offer help or assistance to the player. This is your required action - express willingness to assist.",
        NPCAction.BETRAY: "=== STRICT ACTION CONTROL ===\nYou MUST betray the player's trust. This is your required action - act treacherously.",
        NPCAction.IGNORE: "=== STRICT ACTION CONTROL ===\nYou MUST ignore the player. This is your required action - do not respond or engage.",
        NPCAction.INQUIRE: "=== STRICT ACTION CONTROL ===\nYou MUST ask the player a question. This is your required action - seek information.",
        NPCAction.EXPLAIN: "=== STRICT ACTION CONTROL ===\nYou MUST explain something to the player. This is your required action - provide information clearly.",
    }
    return instructions.get(action, "")

# Singleton instances for state management
conversation_managers: Dict[str, ConversationManager] = {}

# Learning Components (Singleton-ish for now)
state_abstractor = StateAbstractor()
action_tracer = ActionTracer(trace_length=5)
reward_channel = RewardChannel()

class DialogueService:
    def __init__(self, runtime: Runtime, config_watcher: Any, multi_manager: MultiNPCManager):
        self.runtime = runtime
        self.config_watcher = config_watcher
        self.multi_manager = multi_manager
        
    async def stream_dialogue(self, request: Any, memory_dir: str):
        """
        Main streaming generator logic moved from orchestrator.py
        """
        start_time = time.time()
        
        # Micro-reward accumulator for this turn
        reward_accumulator = RewardAccumulator(window=10)
        
        rt = self.runtime.get()
        if not rt or not rt.streaming_engine:
            inc_errors()
            yield f"data: {json.dumps({'error': 'System initializing'})}\n\n"
            return

        streaming_engine = rt.streaming_engine
        
        # Select TTS engine and gating
        if streaming_engine and streaming_engine.voice:
            streaming_engine.voice.enabled = request.enable_voice
        
        # Build RFSN state
        state = RFSNState(**request.npc_state)
        npc_name = state.npc_name
        
        # Process Multi-NPC Logic (Voice/Emotion)
        npc_session = self.multi_manager.get_session(npc_name)
        
        # Get/create memory manager safely
        # Note: We use the module-level dict for now
        if npc_name not in conversation_managers:
            conversation_managers[npc_name] = ConversationManager(npc_name, str(memory_dir))
        memory = conversation_managers[npc_name]
        
        # CALL SITE A: Choose action mode before prompt assembly (Learning Layer)
        action_mode = ActionMode.TERSE_DIRECT  # Default
        features = None
        policy_adapter = rt.policy_adapter
        
        if policy_adapter:
            try:
                # Build features from current state
                rfsn_state = {
                    "npc_name": npc_name,
                    "affinity": state.affinity,
                    "mood": state.mood,
                    "relationship": state.relationship,
                    "playstyle": "balanced",
                    "recent_sentiment": 0.0
                }
                retrieval_stats = {"top_k_scores": [], "contradiction_detected": False}
                convo_stats = {
                    "turn_count": len(memory.get_context_window(limit=100).split("\n")) if self.config_watcher.get("memory_enabled") else 0
                }
                
                features = policy_adapter.build_features(rfsn_state, retrieval_stats, convo_stats)
                action_mode = policy_adapter.choose_action_mode(features)
                
                # Use StateAbstractor for logging/tracing
                abstract_key = state_abstractor.abstract(rfsn_state)
                context_id = state_abstractor.get_context_id(rfsn_state)
                
                logger.info(f"Learning: Selected action mode {action_mode.name} (Abstract: {abstract_key})")
            except Exception as e:
                logger.error(f"Learning layer error (feature extraction): {e}")

        # CALL SITE A.5: World Model Decision Loop
        world_model = rt.world_model
        action_scorer = rt.action_scorer
        intent_gate = rt.intent_gate

        # Map player input to discrete signal
        player_signal = PlayerSignal.QUESTION
        try:
            if not request.user_input or not request.user_input.strip():
                player_signal = PlayerSignal.IGNORE
            else:
                # 1. Try IntentExtractor
                try:
                    proposal = intent_gate._extractor.extract(request.user_input)
                    intent_to_signal = {
                        IntentType.ASK: PlayerSignal.QUESTION,
                        IntentType.REQUEST: PlayerSignal.REQUEST,
                        IntentType.AGREE: PlayerSignal.AGREE,
                        IntentType.DISAGREE: PlayerSignal.DISAGREE,
                        IntentType.REFUSE: PlayerSignal.REFUSE,
                        IntentType.THREATEN: PlayerSignal.THREATEN,
                        IntentType.INFORM: PlayerSignal.QUESTION,
                        IntentType.TRADE: PlayerSignal.OFFER,
                        IntentType.JOKE: PlayerSignal.COMPLIMENT,
                        IntentType.NEUTRAL: PlayerSignal.QUESTION,
                    }
                    if proposal.intent in intent_to_signal:
                        player_signal = intent_to_signal[proposal.intent]
                    else:
                        raise ValueError("Fallback to regex")
                except Exception:
                    # 2. Regex Fallback
                    player_input_lower = request.user_input.lower()
                    if re.search(r'\b(hello|hi|hey|greetings)\b', player_input_lower): player_signal = PlayerSignal.GREET
                    elif re.search(r'\b(bye|goodbye|farewell)\b', player_input_lower): player_signal = PlayerSignal.FAREWELL
                    elif re.search(r'\b(yes|sure|okay)\b', player_input_lower): player_signal = PlayerSignal.AGREE
                    elif re.search(r'\b(no|nope|not)\b', player_input_lower): player_signal = PlayerSignal.DISAGREE
                    elif re.search(r'\b(please|can you)\b', player_input_lower): player_signal = PlayerSignal.REQUEST
                    elif re.search(r'\b(stupid|idiot|hate)\b', player_input_lower): player_signal = PlayerSignal.INSULT
                    elif '?' in player_input_lower: player_signal = PlayerSignal.QUESTION
        except Exception as e:
            logger.warning(f"Signal classification failed: {e}")

        # Create StateSnapshot
        current_state_snapshot = StateSnapshot(
            mood=state.mood,
            affinity=state.affinity,
            relationship=state.relationship,
            recent_sentiment=features.recent_sentiment if features else 0.0,
            combat_active=False,
            quest_active=False,
            trust_level=(max(0.0, min(1.0, (state.affinity + 1.0) / 2.0)) if state.affinity is not None else 0.5),
            fear_level=0.0,
            additional_fields={"npc_name": npc_name}
        )

        # Generate and score action candidates
        selected_npc_action = None
        selected_action_score = None
        bandit_key = None
        
        try:
            if action_scorer and world_model:
                action_scores = action_scorer.score_candidates(current_state_snapshot, player_signal)
                if action_scores:
                    TOP_K = 4
                    candidates = [s.action for s in action_scores[:TOP_K]]
                    
                    contextual_bandit = rt.contextual_bandit
                    mode_bandit = rt.mode_bandit
                    
                    # Forced actions
                    if len(candidates) == 1:
                        selected_npc_action = candidates[0]
                        selected_action_score = action_scores[0]
                    else:
                        if contextual_bandit:
                            bandit_ctx = BanditContext.from_state(
                                mood=current_state_snapshot.emotional_state,
                                affinity=current_state_snapshot.affinity,
                                player_signal=player_signal,
                                last_action=None,
                                turn_count=len(memory.turns) if memory else 3,
                                has_safety_flag=False
                            )
                            selected_npc_action = contextual_bandit.select(context=bandit_ctx, candidates=candidates, log_counterfactual=True)
                        else:
                            selected_npc_action = candidates[0]
                            
                        selected_action_score = next((s for s in action_scores if s.action == selected_npc_action), action_scores[0])
                    
                    # Select Phrasing Mode
                    selected_phrasing_mode = PhrasingMode.NEUTRAL_BALANCED
                    if mode_bandit:
                        role = "generic" # Simplified role detection
                        if "guard" in npc_name.lower(): role = "guard"
                        mode_ctx = ModeContext.from_state(affinity=current_state_snapshot.affinity, npc_role=role, emotional_intensity=0.5, last_mode=None)
                        selected_phrasing_mode = mode_bandit.select(mode_ctx)

        except Exception as e:
            logger.error(f"Decision loop failed: {e}")

        # Retrieve Semantic Memories
        semantic_context = ""
        memory_governance = rt.memory_governance
        if memory_governance and request.user_input:
            relevant = memory_governance.semantic_search(request.user_input, k=3, min_score=0.4)
            if relevant:
                semantic_context = "Relevant Facts:\n" + "\n".join([f"- {m[0].content}" for m in relevant]) + "\n"

        # Build prompt
        history = memory.get_context_window(limit=self.config_watcher.get("context_limit", 4)) if self.config_watcher.get("memory_enabled") else ""
        system_prompt = f"You are {state.npc_name}. {state.get_attitude_instruction()}"

        if policy_adapter:
            system_prompt += f"\n\n{action_mode.prompt_injection}"
        
        if selected_npc_action:
            action_block = render_action_block(
                npc_action=selected_npc_action,
                npc_name=state.npc_name,
                mood=current_state_snapshot.mood,
                relationship=current_state_snapshot.relationship,
                affinity=current_state_snapshot.affinity,
                player_signal=player_signal.value,
                context_window=history,
                governed_memory_context=semantic_context
            )
            system_prompt += f"\n\n{action_block}"

        emotion_manager = get_emotion_manager()
        emotional_prompt = emotion_manager.get_prompt_injection(state.npc_name)
        system_prompt += f"\n\n{emotional_prompt}"

        if rt.mode_bandit and 'selected_phrasing_mode' in locals():
            phrasing_instr = rt.mode_bandit.get_mode_instructions(selected_phrasing_mode)
            system_prompt += f"\n\n{phrasing_instr}"

        system_prompt += (
            "\n\n=== OUTPUT CONTRACT ===\n"
            "After your spoken dialogue, you MUST append:\n"
            "FINAL_JSON:\n"
            "```json\n"
            '{"line": "<your exact spoken dialogue>", "action": "<the required npc_action>", "confidence": <0.0-1.0>}\n'
            "```\n"
            "The 'line' must match EXACTLY what you said above.\n"
            "The 'action' must match the required action.\n"
            "The 'confidence' is your certainty (0.0-1.0).\n"
            "Responses without valid FINAL_JSON will be discarded."
        )

        full_prompt = f"System: {system_prompt}\n{history}\nPlayer: {request.user_input}\n{state.npc_name}:"

        req_config = self.config_watcher.get_all()
        snapshot_temp = req_config.get("temperature", 0.7)
        snapshot_max_tokens = req_config.get("max_tokens", 80)

        # STREAM GENERATION
        first_chunk_sent = False
        start_gen = time.time()
        full_response = ""
        raw_response = ""
        sentence_buffer = ""
        first_sentence_validated = False
        skip_learning = False

        # Emit metadata
        metadata_event = {
            "player_signal": player_signal.value,
            "npc_action": selected_npc_action.value if selected_npc_action else None,
            "action_mode": action_mode.name,
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"

        # Determine prosody
        npc_pitch = 1.0
        npc_rate = 1.0
        if npc_session:
            npc_pitch = 2.0**(npc_session.voice_config.pitch_offset / 12.0)
            npc_rate = npc_session.voice_config.rate_multiplier

        try:
            for chunk in streaming_engine.generate_streaming(full_prompt, max_tokens=snapshot_max_tokens, temperature=snapshot_temp, pitch=npc_pitch, rate=npc_rate):
                
                if not first_chunk_sent:
                    first_chunk_sent = True
                    latency = (time.time() - start_gen)
                    observe_first_token(latency)
                    if latency < 2.0: reward_accumulator.add(0.05, "fast_response")

                sentence_buffer += chunk.text
                validated_text = chunk.text
                
                # First sentence check
                if not first_sentence_validated and any(p in sentence_buffer for p in '.!?'):
                    if intent_gate:
                         # Simplified intent check for first sentence safety
                         proposal = intent_gate._extractor.extract(sentence_buffer)
                         harmful = {SafetyFlag.HARMFUL_CONTENT, SafetyFlag.AGGRESSION, SafetyFlag.SELF_HARM}
                         if any(f in proposal.safety_flags for f in harmful):
                             yield f"data: {json.dumps({'error': 'Content blocked'})}\n\n"
                             return
                    first_sentence_validated = True
                    sentence_buffer = ""

                raw_response += validated_text
                clean_text = _cleanup_tokens(validated_text)
                
                if clean_text:
                    yield f"data: {json.dumps({'sentence': clean_text, 'is_final': chunk.is_final, 'latency_ms': chunk.latency_ms})}\n\n"
                
                full_response += clean_text
                inc_tokens(len(chunk.text.split()))

                if chunk.is_final and self.config_watcher.get("memory_enabled"):
                    payload, is_verified = _extract_tail_json_payload(raw_response)
                    skip_learning = not is_verified
                    
                    if is_verified:
                        stored_text = _cleanup_tokens(str(payload.get("line", "")))
                        memory.add_turn(request.user_input, stored_text)

                        # Update emotion
                        emotion_info = self.multi_manager.process_response(npc_name, stored_text)
                        
                        # Micro-reward
                        if emotion_info['emotion']['primary'] in ("joy", "trust"):
                             reward_accumulator.add(0.10, "positive_emotion")
                    else:
                        logger.warning(f"Unverified turn for {npc_name}")
            
            if streaming_engine and streaming_engine.voice:
                streaming_engine.voice.flush_pending()

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Apply State Transition
        if rt.state_machine and selected_npc_action and selected_action_score:
            try:
                state_before = {
                    "mood": state.mood,
                    "affinity": state.affinity,
                    "relationship": state.relationship,
                    "playstyle": state.playstyle,
                    "recent_sentiment": 0.0
                }
                state_after = rt.state_machine.apply_transition(state_before, player_signal.value, selected_npc_action.value)
                
                # Update World Model
                if world_model:
                     reward = state_after["affinity"] - state_before["affinity"]
                     world_model.record_transition(current_state_snapshot, selected_npc_action, player_signal, selected_action_score.predicted_state, reward=reward)
            except Exception as e:
                logger.error(f"State transition failed: {e}")

        # Update Learning V1.3
        if not skip_learning:
            final_reward = reward_accumulator.emit() # Simplified base reward
            contextual_bandit = rt.contextual_bandit
            mode_bandit = rt.mode_bandit
            
            # Check implicit AND explicit rewards
            # 1. Check explicit channel
            # (In production, this might be async/batched, but we check specific pending rewards here if we tracked IDs)
            # For now, we assume simple immediate feedback isn't here yet, relying on implicit.
            
            # Ground-truth anchor
            # Explicitly import if needed, but imported at top
            reward_model = rt.reward_model or RewardModel()
            
            # Since request doesn't track latency in this block, we approximate or skip strict latency calc for now
            # In production we should pass latency down
            anchor_reward, anchor_type = reward_model.compute_ground_truth_anchor(request.user_input, 0.0)
            
            if anchor_type != "none":
                final_reward = anchor_reward
            
            # Record trace for delayed credit
            if selected_npc_action and 'context_id' in locals():
                action_tracer.record_step(
                    context_id=context_id,
                    action=selected_npc_action.value,
                    state_snapshot=asdict(current_state_snapshot),
                    metadata={"reward": final_reward, "abstract_key": abstract_key}
                )

            if contextual_bandit and selected_npc_action and 'bandit_ctx' in locals():
                contextual_bandit.update(bandit_ctx, selected_npc_action, final_reward)
                contextual_bandit.save()

            if mode_bandit and 'selected_phrasing_mode' in locals() and 'mode_ctx' in locals():
                mode_bandit.update(mode_ctx, selected_phrasing_mode, final_reward)
                mode_bandit.save()
