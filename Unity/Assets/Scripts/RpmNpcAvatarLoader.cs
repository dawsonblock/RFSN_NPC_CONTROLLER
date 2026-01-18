using System;
using UnityEngine;

// RPM loader namespaces are provided by the Ready Player Me Core package.
using ReadyPlayerMe.AvatarLoader;

public class RpmNpcAvatarLoader : MonoBehaviour
{
    [Header("RPM Avatar")]
    [Tooltip("Full Ready Player Me avatar URL (ends with .glb) or short URL the SDK accepts.")]
    public string avatarUrl;

    [Tooltip("Where the loaded avatar GameObject will be parented.")]
    public Transform avatarAnchor;

    [Header("Optional")]
    public bool destroyPreviousAvatar = true;

    private GameObject currentAvatar;
    private AvatarObjectLoader avatarLoader;

    public void Load()
    {
        if (avatarAnchor == null)
        {
            Debug.LogError("avatarAnchor not assigned.");
            return;
        }

        if (destroyPreviousAvatar && currentAvatar != null)
        {
            Destroy(currentAvatar);
            currentAvatar = null;
        }

        avatarLoader ??= new AvatarObjectLoader();

        avatarLoader.OnCompleted += OnAvatarLoaded;
        avatarLoader.OnFailed += OnAvatarFailed;

        avatarLoader.LoadAvatar(avatarUrl);
    }

    private void OnAvatarLoaded(GameObject avatar)
    {
        avatarLoader.OnCompleted -= OnAvatarLoaded;
        avatarLoader.OnFailed -= OnAvatarFailed;

        currentAvatar = avatar;
        currentAvatar.transform.SetParent(avatarAnchor, false);

        // Make sure there is an Animator we can drive.
        // RPM avatars are Humanoid-ready; you'll typically add/drive a controller on the root Animator.
        Debug.Log($"RPM avatar loaded: {avatar.name}");
    }

    private void OnAvatarFailed(object sender, FailureEventArgs args)
    {
        avatarLoader.OnCompleted -= OnAvatarLoaded;
        avatarLoader.OnFailed -= OnAvatarFailed;

        Debug.LogError($"RPM avatar load failed: {args.Type} - {args.Message}");
    }
}
