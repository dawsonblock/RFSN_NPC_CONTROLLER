# RFSN Unity Integration

This package provides Unity scripts to connect Ready Player Me avatars with the RFSN Orchestrator backend.

## Installation

### 1. Import Ready Player Me Core

1. Unity → Window → Package Manager → + → Add package from git URL
2. Paste: `https://github.com/readyplayerme/rpm-unity-sdk-core.git`
3. In Package Manager, select Ready Player Me Core and import the AvatarLoadingSamples sample
4. Test the QuickStart scene to verify RPM is working

### 2. Copy Scripts

Copy all scripts from `Assets/Scripts/` to your Unity project:
- `RpmNpcAvatarLoader.cs`
- `RfsnNpcClient.cs`
- `NpcInteractionTrigger.cs`
- `NpcSpawner.cs`

### 3. Create NPC Prefab

1. Create empty GameObject: `NPC_RPM_Root`
2. Add components:
   - `CapsuleCollider`
   - `NavMeshAgent`
   - `Animator`
   - `RpmNpcAvatarLoader` (from this package)
   - `RfsnNpcClient` (from this package)
   - `NpcInteractionTrigger` (from this package)
3. Create child GameObject: `AvatarAnchor` (empty)
4. In `RpmNpcAvatarLoader`, drag `AvatarAnchor` into the `avatarAnchor` field
5. In `RfsnNpcClient`, assign the `animator` and `agent` fields
6. In `NpcInteractionTrigger`, assign the `npcClient` field
7. Save as prefab: `Assets/Prefabs/NPC_RPM_Root.prefab`

### 4. Create Animator Controller

1. Create `Assets/Animations/Npc.controller`
2. Add triggers: `Greet`, `Warn`, `Idle`, `Flee`, `Attack`, `Trade`, `Talk`
3. Wire them to animations (use placeholder idle animations at first)
4. Assign this controller to the NPC's Animator component

### 5. Setup Spawner

1. Create empty GameObject: `NpcSpawner`
2. Add `NpcSpawner` component
3. Assign fields:
   - `npcPrefab`: Your `NPC_RPM_Root` prefab
   - `playerTransform`: Your player GameObject
   - `npcConfigs`: Array of NPC configurations (see below)

### 6. Configure NPCs

In the `NpcSpawner` component, add NPC configurations:

| Field | Description | Example |
|-------|-------------|---------|
| `npcName` | Display name | "Lydia" |
| `npcId` | Unique ID | "npc_lydia_001" |
| `avatarUrl` | RPM avatar URL | "https://models.readyplayer.me/64f0..." |

Get avatar URLs from:
- https://readyplayer.me/ (create avatars, copy URL)
- Your own RPM subdomain

### 7. Test

1. Play scene
2. Walk near an NPC (within `interactionDistance`)
3. Press `E` (or configured key)
4. NPC should:
   - Load RPM avatar
   - Send "Hello" to orchestrator
   - Receive meta event with `npc_action`
   - Trigger animation (e.g., "Greet")
   - Stream dialogue sentences to console

## SSE Payload Schema

Your orchestrator emits Server-Sent Events in this format:

### Meta Event (first)
```json
{
  "player_signal": "QUESTION",
  "bandit_key": "...",
  "npc_action": "GREET",
  "action_mode": "TERSE_DIRECT"
}
```

### Sentence Events (streamed)
```json
{
  "sentence": "Hey. You new here?",
  "is_final": false,
  "latency_ms": 123
}
```

## NPC Action Mapping

The `RfsnNpcClient` maps orchestrator `npc_action` values to Animator triggers:

| npc_action | Animator Trigger |
|------------|------------------|
| `GREET` | `Greet` |
| `WARN` | `Warn` |
| `IDLE` | `Idle` |
| `FLEE` | `Flee` |
| `ATTACK` | `Attack` |
| `TRADE` | `Trade` |
| (default) | `Talk` |

## Customization

### Change Orchestrator URL

Edit `RfsnNpcClient.orchestratorStreamUrl` in the Inspector or code:
```csharp
public string orchestratorStreamUrl = "http://127.0.0.1:8000/api/dialogue/stream";
```

### Add Custom Animations

1. Add trigger to Animator Controller
2. Add case to `ApplyNpcAction()` in `RfsnNpcClient.cs`:
```csharp
case "CUSTOM_ACTION":
    animator.SetTrigger("CustomTrigger");
    break;
```

### Add Subtitles

In `RfsnNpcClient.cs`, replace the Debug.Log in the sentence handling:
```csharp
// TODO: show subtitles / bubble text / send to TTS
subtitleUI.text = s.sentence;
```

### Add TTS Audio

The orchestrator already streams TTS audio. To play it in Unity:

1. Add `AudioSource` component to NPC prefab
2. Modify `RfsnNpcClient` to handle audio events (if your orchestrator emits them)
3. Or use a separate TTS service in Unity

## Troubleshooting

### Avatar not loading
- Check console for RPM errors
- Verify `avatarUrl` is valid (ends with `.glb`)
- Ensure `AvatarAnchor` is assigned

### No animation triggers
- Verify Animator Controller is assigned
- Check triggers exist in controller
- Verify `npc_action` values match cases in `ApplyNpcAction()`

### SSE stream not connecting
- Check `orchestratorStreamUrl` is correct
- Verify orchestrator is running
- Check browser console/network tab for errors
- Ensure CORS is configured on orchestrator

### Interaction not working
- Verify `playerTransform` is assigned
- Check `interactionDistance` (default 3.0 units)
- Verify `interactionKey` (default E)
- Check NPC has `NpcInteractionTrigger` component

## Next Steps

1. **Subtitles UI**: Create a subtitle bubble above NPC's head
2. **Dialogue UI**: Build a full dialogue system with player choices
3. **TTS Integration**: Connect orchestrator's TTS audio to Unity's AudioSource
4. **Lip Sync**: Integrate visemes/lip sync (e.g., OVRLipSync)
5. **Movement**: Implement NavMeshAgent destination setting for actions like FLEE
6. **Multi-NPC**: Test multiple NPCs with different avatars and personalities

## Architecture

```
Player (presses E)
  → NpcInteractionTrigger detects proximity
  → RfsnNpcClient.SendPlayerUtterance("Hello")
  → POST to /api/dialogue/stream
  → SSE stream:
    1. Meta event: {"npc_action":"GREET", ...}
    2. Sentence events: {"sentence":"Hey...", ...}
  → RfsnNpcClient.ApplyNpcAction("GREET")
  → Animator.SetTrigger("Greet")
  → NPC plays Greet animation
  → Sentences logged/shown as subtitles
```

## License

Part of RFSN Orchestrator project.
