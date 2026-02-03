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


def _build_keyframes(start, end, fade, strength_model, low):
    """Build HookKeyframeGroup based on fade settings.

    strength_mult is a multiplier on base strength_model.
    """
    hook_kf = comfy.hooks.HookKeyframeGroup()

    if strength_model == 0:
        return hook_kf

    low_mult = low / strength_model if strength_model > 0 else 0.0

    if fade == "none":
        # Constant strength within [start, end]
        if start > 0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=0.0, guarantee_steps=1,
            ))
        hook_kf.add(comfy.hooks.HookKeyframe(
            strength=1.0, start_percent=start, guarantee_steps=1,
        ))
        if end < 1.0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=end,
            ))

    elif fade == "fade out":
        # 1.0 → low_mult over [start, end]
        if start > 0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=0.0, guarantee_steps=1,
            ))
        for i in range(INTERPOLATION_STEPS + 1):
            t = i / INTERPOLATION_STEPS
            pct = start + (end - start) * t
            mult = 1.0 + (low_mult - 1.0) * t
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=mult, start_percent=pct,
                guarantee_steps=1 if i == 0 else 0,
            ))
        if end < 1.0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=end + 0.001,
            ))

    elif fade == "fade in":
        # low_mult → 1.0 over [start, end]
        if start > 0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=0.0, guarantee_steps=1,
            ))
        for i in range(INTERPOLATION_STEPS + 1):
            t = i / INTERPOLATION_STEPS
            pct = start + (end - start) * t
            mult = low_mult + (1.0 - low_mult) * t
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=mult, start_percent=pct,
                guarantee_steps=1 if i == 0 else 0,
            ))
        if end < 1.0:
            hook_kf.add(comfy.hooks.HookKeyframe(
                strength=0.0, start_percent=end + 0.001,
            ))

    return hook_kf


class LoraLoaderAdvancedNode:
    """All-in-one LoRA loader with step-based strength scheduling.

    Uses ComfyUI Hook Keyframe system for per-step MODEL strength control.
    CLIP receives fixed strength (no fade).
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
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "fade": (
                    ["none", "fade out", "fade in"],
                    {"default": "none"},
                ),
                "low": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
    RETURN_NAMES = ("model", "clip", "hooks")
    FUNCTION = "load_lora"
    CATEGORY = "IXIWORKS/LoRA"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip,
                  start, end, fade, low):
        if strength_model == 0 and strength_clip == 0:
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

        # Apply LoRA to CLIP with fixed strength (traditional method)
        new_clip = clip
        if strength_clip != 0:
            _, new_clip = comfy.sd.load_lora_for_models(
                None, clip, lora, 0, strength_clip,
            )

        # Create Hook for MODEL with keyframe scheduling
        hooks = None
        if strength_model != 0:
            hooks = comfy.hooks.create_hook_lora(
                lora=lora, strength_model=strength_model, strength_clip=0,
            )
            hook_kf = _build_keyframes(start, end, fade, strength_model, low)
            for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight):
                hook.hook_keyframe = hook_kf

        logger.info(
            f"[IXIWORKS] LoRA Advanced: '{lora_name}' "
            f"model={strength_model} clip={strength_clip} "
            f"fade={fade} start={start} end={end} low={low}"
        )

        return (model, new_clip, hooks)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderAdvanced": LoraLoaderAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderAdvanced": "LoRA Loader Advanced",
}
