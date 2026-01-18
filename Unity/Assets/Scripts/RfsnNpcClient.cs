using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AI;

// Exact JSON schema matching your orchestrator's SSE output
[Serializable]
public class OrchestratorMeta
{
    public string player_signal;  // e.g. "QUESTION", "GREET"
    public string bandit_key;     // learning state key
    public string npc_action;     // e.g. "GREET", "WARN", "FLEE"
    public string action_mode;    // e.g. "TERSE_DIRECT", "VERBOSE"
}

[Serializable]
public class OrchestratorSentence
{
    public string sentence;
    public bool is_final;
    public float latency_ms;
}

public class RfsnNpcClient : MonoBehaviour
{
    [Header("Orchestrator")]
    public string orchestratorStreamUrl = "http://127.0.0.1:8000/api/dialogue/stream";

    [Header("Unity hooks")]
    public Animator animator;
    public NavMeshAgent agent;

    [Header("NPC identity")]
    public string npcName = "NPC";
    public string npcId = "npc_001";

    private static readonly HttpClient http = new HttpClient();
    private CancellationTokenSource currentCts;

    // Call this from your interaction system when the player speaks/chooses a line.
    public void SendPlayerUtterance(string playerText)
    {
        currentCts?.Cancel();
        currentCts = new CancellationTokenSource();
        _ = StreamTurn(playerText, currentCts.Token);
    }

    private async Task StreamTurn(string playerText, CancellationToken ct)
    {
        // Build request matching your DialogueRequest schema
        var payload = new
        {
            user_input = playerText,
            npc_state = new
            {
                npc_name = npcName,
                npc_id = npcId,
                affinity = 0.0f,  // You can track this client-side
                mood = "Neutral",
                relationship = "Stranger"
            },
            tts_engine = "piper"  // or "xvasynth"
        };

        string json = JsonUtility.ToJson(new RequestWrapper(payload));
        using var req = new HttpRequestMessage(HttpMethod.Post, orchestratorStreamUrl);
        req.Content = new StringContent(json, Encoding.UTF8, "application/json");

        try
        {
            using HttpResponseMessage resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            resp.EnsureSuccessStatusCode();

            using Stream stream = await resp.Content.ReadAsStreamAsync(ct);
            using var reader = new StreamReader(stream);

            bool gotMeta = false;

            while (!reader.EndOfStream && !ct.IsCancellationRequested)
            {
                string line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line)) continue;

                // SSE lines look like: "data: {...json...}"
                if (!line.StartsWith("data:")) continue;

                string data = line.Substring(5).Trim();
                if (string.IsNullOrWhiteSpace(data)) continue;

                // Try meta first (has npc_action field)
                if (!gotMeta && data.Contains("\"npc_action\""))
                {
                    gotMeta = true;
                    var meta = JsonUtility.FromJson<OrchestratorMeta>(data);
                    ApplyNpcAction(meta.npc_action);
                    Debug.Log($"[Meta] action={meta.npc_action}, mode={meta.action_mode}, signal={meta.player_signal}");
                    continue;
                }

                // Sentence event: has "sentence" field
                if (data.Contains("\"sentence\""))
                {
                    var s = JsonUtility.FromJson<OrchestratorSentence>(data);
                    if (!string.IsNullOrWhiteSpace(s.sentence))
                    {
                        // TODO: show subtitles / bubble text / send to TTS
                        Debug.Log($"[{npcName}] {s.sentence}");
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when starting new turn
        }
        catch (Exception e)
        {
            Debug.LogError($"SSE stream error: {e}");
        }
    }

    private void ApplyNpcAction(string npcAction)
    {
        if (string.IsNullOrWhiteSpace(npcAction) || animator == null) return;

        // Map your orchestrator's NPCAction enum values to animator triggers
        switch (npcAction)
        {
            case "GREET":
                animator.SetTrigger("Greet");
                break;

            case "WARN":
                animator.SetTrigger("Warn");
                break;

            case "IDLE":
                animator.SetTrigger("Idle");
                break;

            case "FLEE":
                animator.SetTrigger("Flee");
                if (agent != null)
                {
                    // Move away from player (you'd need player position reference)
                    // agent.SetDestination(...);
                }
                break;

            case "ATTACK":
                animator.SetTrigger("Attack");
                break;

            case "TRADE":
                animator.SetTrigger("Trade");
                break;

            default:
                animator.SetTrigger("Talk");
                break;
        }
    }

    // JsonUtility can't serialize anonymous objects directly; wrap it.
    [Serializable]
    private class RequestWrapper
    {
        public string user_input;
        public NpcState npc_state;
        public string tts_engine;

        public RequestWrapper(object o)
        {
            var t = o.GetType();
            user_input = (string)t.GetProperty("user_input")!.GetValue(o);
            npc_state = new NpcState(t.GetProperty("npc_state")!.GetValue(o));
            tts_engine = (string)t.GetProperty("tts_engine")!.GetValue(o);
        }
    }

    [Serializable]
    private class NpcState
    {
        public string npc_name;
        public string npc_id;
        public float affinity;
        public string mood;
        public string relationship;

        public NpcState(object o)
        {
            var t = o.GetType();
            npc_name = (string)t.GetProperty("npc_name")!.GetValue(o);
            npc_id = (string)t.GetProperty("npc_id")!.GetValue(o);
            affinity = (float)(double)t.GetProperty("affinity")!.GetValue(o);
            mood = (string)t.GetProperty("mood")!.GetValue(o);
            relationship = (string)t.GetProperty("relationship")!.GetValue(o);
        }
    }
}
