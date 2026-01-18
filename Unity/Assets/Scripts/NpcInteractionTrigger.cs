using UnityEngine;

public class NpcInteractionTrigger : MonoBehaviour
{
    [Header("Interaction Settings")]
    [Tooltip("The NPC client script to trigger.")]
    public RfsnNpcClient npcClient;

    [Tooltip("The player transform to check distance from.")]
    public Transform playerTransform;

    [Tooltip("Maximum interaction distance in units.")]
    public float interactionDistance = 3.0f;

    [Tooltip("UI GameObject to show when in range (e.g., 'Press E to talk').")]
    public GameObject interactionPrompt;

    [Header("Optional")]
    [Tooltip("Key to press to interact.")]
    public KeyCode interactionKey = KeyCode.E;

    private bool isInRange;

    private void Update()
    {
        if (playerTransform == null || npcClient == null) return;

        float distance = Vector3.Distance(transform.position, playerTransform.position);
        bool wasInRange = isInRange;
        isInRange = distance <= interactionDistance;

        // Show/hide prompt
        if (interactionPrompt != null)
        {
            interactionPrompt.SetActive(isInRange);
        }

        // Handle interaction key press
        if (isInRange && Input.GetKeyDown(interactionKey))
        {
            // For now, trigger a simple greeting
            // In a real game, you'd open a dialogue UI or input field
            npcClient.SendPlayerUtterance("Hello");
        }
    }

    private void OnDrawGizmosSelected()
    {
        // Draw interaction radius in editor
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, interactionDistance);
    }
}
