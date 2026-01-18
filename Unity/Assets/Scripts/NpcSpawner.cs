using UnityEngine;

public class NpcSpawner : MonoBehaviour
{
    [Header("NPC Prefab")]
    [Tooltip("The NPC_RPM_Root prefab with RpmNpcAvatarLoader and RfsnNpcClient.")]
    public GameObject npcPrefab;

    [Header("Avatar Configuration")]
    [Tooltip("Map NPC names to their RPM avatar URLs.")]
    public NpcAvatarConfig[] npcConfigs;

    [Header("Spawn Settings")]
    [Tooltip("Where to spawn NPCs (optional, uses current position if null).")]
    public Transform spawnPoint;

    [Tooltip("Player transform for interaction distance checks.")]
    public Transform playerTransform;

    void Start()
    {
        SpawnAllNpcs();
    }

    public void SpawnAllNpcs()
    {
        if (npcPrefab == null)
        {
            Debug.LogError("NPC prefab not assigned!");
            return;
        }

        Vector3 spawnPos = spawnPoint != null ? spawnPoint.position : transform.position;

        foreach (var config in npcConfigs)
        {
            SpawnNpc(config, spawnPos);
        }
    }

    private void SpawnNpc(NpcAvatarConfig config, Vector3 position)
    {
        // Instantiate NPC prefab
        GameObject npcInstance = Instantiate(npcPrefab, position, Quaternion.identity);

        // Configure NPC identity
        var npcClient = npcInstance.GetComponent<RfsnNpcClient>();
        if (npcClient != null)
        {
            npcClient.npcName = config.npcName;
            npcClient.npcId = config.npcId;
        }

        // Configure RPM avatar loader
        var avatarLoader = npcInstance.GetComponent<RpmNpcAvatarLoader>();
        if (avatarLoader != null)
        {
            avatarLoader.avatarUrl = config.avatarUrl;
            avatarLoader.Load();
        }

        // Configure interaction trigger
        var interactionTrigger = npcInstance.GetComponent<NpcInteractionTrigger>();
        if (interactionTrigger != null && playerTransform != null)
        {
            interactionTrigger.playerTransform = playerTransform;
        }

        Debug.Log($"Spawned NPC: {config.npcName}");
    }
}

[Serializable]
public class NpcAvatarConfig
{
    public string npcName;
    public string npcId;
    [Tooltip("Ready Player Me avatar URL (ends with .glb)")]
    public string avatarUrl;
}
