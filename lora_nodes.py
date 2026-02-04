"""
LoRA Loader Advanced Node for ComfyUI
Step-based LoRA strength scheduling using ComfyUI Hook Keyframe system
"""

import logging

import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

INTERPOLATION_STEPS = 10


def _build_keyframes(start_at, end_at, strength_start, strength_end):
    """Build HookKeyframeGroup based on strength_start/end settings.

    Auto-detects fade direction from strength values.
    """
    hook_kf = comfy.hooks.HookKeyframeGroup()

    if strength_start == 0 and strength_end == 0:
        return hook_kf

    # Calculate multipliers (relative to strength_start for hook system)
    base_strength = strength_start if strength_start > 0 else strength_end
    start_mult = strength_start / base_strength if base_strength > 0 else 0.0
    end_mult = strength_end / base_strength if base_strength > 0 else 0.0

    # Auto-detect: if start == end, no fade (constant)
    use_fade = (strength_start != strength_end)

    if not use_fade:
        # Constant strength within [start_at, end_at]
        if start_at > 0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=0.0, guarantee_steps=1,
            ))
        hook_kf.add(comfy.hooks.HookKeyframe(
            strength=1.0, start_percent=start_at, guarantee_steps=1,
        ))
        if end_at < 1.0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=end_at,
            ))
    else:
        # Linear interpolation from start_mult to end_mult
        if start_at > 0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=0.0, guarantee_steps=1,
            ))
        for i in range(INTERPOLATION_STEPS + 1):
            t = i / INTERPOLATION_STEPS
            pct = start_at + (end_at - start_at) * t
            mult = start_mult + (end_mult - start_mult) * t
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=mult, start_percent=pct,
                guarantee_steps=1 if i == 0 else 0,
            ))
        if end_at < 1.0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=end_at + 0.001,
            ))

    return hook_kf


class LoraLoaderAdvancedNode:
    """All-in-one LoRA loader with step-based strength scheduling.

    Uses ComfyUI Hook Keyframe system for per-step MODEL strength control.
    CLIP receives strength_start value (fixed, no fade).
    HOOKS output must be connected to a conditioning node
    (e.g. PairConditioningSetProperties).
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_start": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "strength_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "start_at": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "end_at": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    RETURN_NAMES = ("model", "clip", "hooks")
    FUNCTION = "load_lora"
    CATEGORY = "IXIWORKS/LoRA"

    def load_lora(self, model, clip, lora_name, strength_start, strength_end,
                  start_at, end_at):
        if strength_start == 0 and strength_end == 0:
            return (model, clip, None)

        # Load LoRA file (with caching)
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        # Apply LoRA to CLIP with strength_start (fixed, no fade)
        new_clip = clip
        if strength_start != 0:
            _, new_clip = comfy.sd.load_lora_for_models(
                None, clip, lora, 0, strength_start,
            )

        # Create Hook for MODEL with keyframe scheduling
        hooks = None
        base_strength = max(strength_start, strength_end)
        if base_strength != 0:
            hooks = comfy.hooks.create_hook_lora(
                lora=lora, strength_model=base_strength, strength_clip=0,
            )
            hook_kf = _build_keyframes(start_at, end_at, strength_start, strength_end)
            for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight):
                hook.hook_keyframe = hook_kf

        # Auto-detect fade direction for logging
        fade_dir = "none"
        if strength_start > strength_end:
            fade_dir = "fade out"
        elif strength_start < strength_end:
            fade_dir = "fade in"

        logger.info(
            f"[IXIWORKS] LoRA Step: '{lora_name}' "
            f"strength={strength_start}→{strength_end} ({fade_dir}) "
            f"range={start_at}→{end_at}"
        )

        return (model, new_clip, hooks)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderAdvanced": LoraLoaderAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderAdvanced": "LoRA Step Loader",
}
