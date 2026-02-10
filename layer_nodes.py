"""
Layer Separation Nodes for ComfyUI
Attention-based layer decomposition for animatics production
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

logger = logging.getLogger(__name__)


# =============================================================================
# Attention Capture Infrastructure
# =============================================================================

@dataclass
class AttentionCapture:
    """Captured attention data for a specific layer and step"""
    layer_idx: int
    step: int
    attention_weights: Optional[torch.Tensor] = None  # (batch, heads, seq, seq)


class AttentionHookManager:
    """
    Manages attention hooks for extracting Q, K and computing attention weights.

    Designed for single-stream DiT models (like Z-Image-Turbo) where text and
    image tokens are concatenated in one sequence.
    """

    def __init__(
        self,
        target_layers: Optional[List[int]] = None,
        target_steps: Optional[List[int]] = None,
        text_token_count: int = 77,
        visual_token_count: int = 256,
    ):
        """
        Args:
            target_layers: Layer indices to capture (default: [10, 15, 20])
            target_steps: Sampling steps to capture (default: capture all)
            text_token_count: Number of text tokens in sequence
            visual_token_count: Number of visual semantic tokens
        """
        self.target_layers = target_layers or [10, 15, 20]
        self.target_steps = target_steps  # None = capture all steps
        self.text_token_count = text_token_count
        self.visual_token_count = visual_token_count

        self.captures: Dict[int, List[AttentionCapture]] = {}
        self.hooks: List = []
        self.current_step = 0
        self._enabled = True

    def _get_image_start_idx(self) -> int:
        """Get the starting index of image tokens in the sequence"""
        return self.text_token_count + self.visual_token_count

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for capturing Q, K and computing attention weights.

        Supports two attention module patterns:
        1. Separate: to_q, to_k, to_v (Stable Diffusion style)
        2. Fused QKV: qkv single linear (Z-Image JointAttention style)
        """
        manager = self

        def hook_fn(module, input, output):
            if not manager._enabled:
                return

            # Check if we should capture this step
            if manager.target_steps is not None:
                if manager.current_step not in manager.target_steps:
                    return

            try:
                hidden_states = input[0]

                # ===== Pattern 1: Fused QKV (Z-Image JointAttention) =====
                if hasattr(module, 'qkv'):
                    qkv_out = module.qkv(hidden_states)
                    # qkv_out shape: (batch, seq, 3 * hidden_dim)
                    # Split into Q, K, V
                    hidden_dim = qkv_out.shape[-1] // 3
                    query, key, value = qkv_out.chunk(3, dim=-1)

                    # Determine num_heads from q_norm or known head_dim
                    if hasattr(module, 'q_norm') and hasattr(module.q_norm, 'weight'):
                        head_dim = module.q_norm.weight.shape[0]
                        num_heads = hidden_dim // head_dim
                    else:
                        num_heads = getattr(module, 'num_heads', 30)
                        head_dim = hidden_dim // num_heads

                    # Reshape for multi-head: (batch, seq, dim) → (batch, heads, seq, head_dim)
                    batch_size, seq_len = query.shape[0], query.shape[1]
                    q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

                    # Apply QK-Norm (RMSNorm per head)
                    if hasattr(module, 'q_norm'):
                        q = module.q_norm(q)
                    if hasattr(module, 'k_norm'):
                        k = module.k_norm(k)

                # ===== Pattern 2: Separate Q, K (SD style) =====
                elif hasattr(module, 'to_q') and hasattr(module, 'to_k'):
                    query = module.to_q(hidden_states)
                    key = module.to_k(hidden_states)

                    if hasattr(module, 'norm_q'):
                        query = module.norm_q(query)
                    if hasattr(module, 'norm_k'):
                        key = module.norm_k(key)

                    num_heads = getattr(module, 'num_heads', 32)
                    head_dim = query.shape[-1] // num_heads
                    batch_size, seq_len = query.shape[0], query.shape[1]

                    q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

                else:
                    logger.warning(f"[AttentionHook] Layer {layer_idx}: unknown attention pattern")
                    return

                # Compute attention weights: softmax(Q @ K^T / sqrt(d))
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)

                # Store capture
                if layer_idx not in manager.captures:
                    manager.captures[layer_idx] = []

                manager.captures[layer_idx].append(AttentionCapture(
                    layer_idx=layer_idx,
                    step=manager.current_step,
                    attention_weights=attn_weights.detach().cpu()
                ))

                logger.debug(
                    f"[AttentionHook] Captured layer {layer_idx}, step {manager.current_step}, "
                    f"attn shape: {tuple(attn_weights.shape)}"
                )

            except Exception as e:
                logger.warning(f"[AttentionHook] Capture failed at layer {layer_idx}: {e}")

        return hook_fn

    def register_hooks(self, model):
        """
        Register hooks on the model's attention layers.

        Args:
            model: The diffusion model (expected to have transformer blocks)
        """
        # Try to find transformer blocks
        # Different models may have different structures
        blocks = None

        # ComfyUI model wrapper
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diff_model = model.model.diffusion_model
            for attr in ['blocks', 'transformer_blocks', 'layers']:
                if hasattr(diff_model, attr):
                    blocks = getattr(diff_model, attr)
                    break

        # Direct model access
        if blocks is None:
            for attr in ['blocks', 'transformer_blocks', 'layers']:
                if hasattr(model, attr):
                    blocks = getattr(model, attr)
                    break

        if blocks is None:
            logger.warning("[AttentionHook] Could not find transformer blocks in model")
            return False

        # Register hooks on target layers
        for idx in self.target_layers:
            if idx >= len(blocks):
                logger.warning(f"[AttentionHook] Layer {idx} out of range (max: {len(blocks)-1})")
                continue

            block = blocks[idx]

            # Find attention module in block
            attn_module = None
            if hasattr(block, 'attn'):
                attn_module = block.attn
            elif hasattr(block, 'self_attn'):
                attn_module = block.self_attn
            elif hasattr(block, 'attention'):
                attn_module = block.attention

            if attn_module is None:
                logger.warning(f"[AttentionHook] Could not find attention module in block {idx}")
                continue

            hook = attn_module.register_forward_hook(self._create_hook(idx))
            self.hooks.append(hook)
            logger.info(f"[AttentionHook] Registered hook on layer {idx}")

        return len(self.hooks) > 0

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info(f"[AttentionHook] Removed all hooks")

    def set_step(self, step: int):
        """Set current sampling step"""
        self.current_step = step

    def enable(self):
        """Enable attention capture"""
        self._enabled = True

    def disable(self):
        """Disable attention capture"""
        self._enabled = False

    def clear_captures(self):
        """Clear all captured data"""
        self.captures.clear()

    def get_token_heatmap(
        self,
        token_indices: List[int],
        layer_idx: int = 15,
        step: int = -1,
        image_size: Tuple[int, int] = (64, 64),
        aggregate_heads: str = "mean"
    ) -> torch.Tensor:
        """
        Extract attention heatmap for specific tokens.

        Args:
            token_indices: Text token indices to extract attention for
            layer_idx: Which layer's attention to use
            step: Which sampling step (-1 for last)
            image_size: Output heatmap size (H, W)
            aggregate_heads: How to aggregate heads ("mean", "max")

        Returns:
            Heatmap tensor of shape (H, W)
        """
        if layer_idx not in self.captures:
            raise ValueError(f"Layer {layer_idx} not captured")

        captures = self.captures[layer_idx]
        if not captures:
            raise ValueError(f"No captures for layer {layer_idx}")

        # Select step
        capture = captures[step] if step >= 0 and step < len(captures) else captures[-1]

        if capture.attention_weights is None:
            raise ValueError("Attention weights not captured")

        attn = capture.attention_weights  # (batch, heads, seq_q, seq_k)

        # Extract text→image attention
        image_start = self._get_image_start_idx()

        # Get attention from specified tokens to image tokens
        # attn shape: (batch, heads, seq, seq)
        text_to_image = attn[:, :, token_indices, image_start:]

        # Aggregate across batch, heads, and tokens
        if aggregate_heads == "mean":
            heatmap = text_to_image.mean(dim=(0, 1, 2))
        elif aggregate_heads == "max":
            heatmap = text_to_image.max(dim=1)[0].max(dim=0)[0].mean(dim=0)
        else:
            heatmap = text_to_image.mean(dim=(0, 1, 2))

        # Reshape to spatial dimensions
        # Assuming square image tokens
        img_tokens = int(math.sqrt(heatmap.shape[0]))
        if img_tokens * img_tokens != heatmap.shape[0]:
            logger.warning(f"[AttentionHook] Non-square image tokens: {heatmap.shape[0]}")
            img_tokens = int(math.sqrt(heatmap.shape[0]))

        heatmap = heatmap[:img_tokens*img_tokens].view(img_tokens, img_tokens)

        # Resize to target size
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0).float(),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def get_all_token_heatmaps(
        self,
        token_groups: Dict[str, List[int]],
        layer_idx: int = 15,
        step: int = -1,
        image_size: Tuple[int, int] = (64, 64)
    ) -> Dict[str, torch.Tensor]:
        """
        Extract heatmaps for multiple token groups.

        Args:
            token_groups: Dict mapping names to token indices, e.g. {"Mina": [3,4], "Robot": [8]}
            layer_idx: Which layer's attention to use
            step: Which sampling step
            image_size: Output heatmap size

        Returns:
            Dict mapping names to heatmap tensors
        """
        heatmaps = {}
        for name, indices in token_groups.items():
            try:
                heatmaps[name] = self.get_token_heatmap(
                    token_indices=indices,
                    layer_idx=layer_idx,
                    step=step,
                    image_size=image_size
                )
            except Exception as e:
                logger.warning(f"[AttentionHook] Failed to get heatmap for '{name}': {e}")

        return heatmaps


# =============================================================================
# KSamplerLayered Node
# =============================================================================

class KSamplerLayered:
    """
    KSampler with integrated attention extraction for layer separation.

    Combines sampling, progress tracking, and attention capture in a single node
    to avoid hook conflicts.

    Outputs:
    - latent: Standard latent output for VAE decode
    - attention_maps: Dict of token heatmaps for mask generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "extract_tokens": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Mina,Robot (comma-separated)"
                }),
                "attention_layers": ("STRING", {
                    "default": "15",
                    "multiline": False,
                    "placeholder": "10,15,20"
                }),
                "redis_url": ("STRING", {"default": ""}),
                "job_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT", "ATTENTION_MAPS")
    RETURN_NAMES = ("latent", "attention_maps")
    FUNCTION = "sample"
    CATEGORY = "IXIWORKS/Layer"

    def __init__(self):
        self.attention_manager = None

    def _parse_attention_layers(self, layers_str: str) -> List[int]:
        """Parse comma-separated layer indices"""
        if not layers_str.strip():
            return [15]  # Default to middle layer

        try:
            return [int(x.strip()) for x in layers_str.split(",") if x.strip()]
        except ValueError:
            logger.warning(f"[KSamplerLayered] Invalid attention_layers: {layers_str}, using default")
            return [15]

    def _parse_extract_tokens(self, tokens_str: str) -> List[str]:
        """Parse comma-separated token names"""
        if not tokens_str.strip():
            return []
        return [x.strip() for x in tokens_str.split(",") if x.strip()]

    def _find_token_indices(self, text: str, target_tokens: List[str]) -> Dict[str, List[int]]:
        """
        Find token indices for target words in the prompt.

        This is a simplified version - in production, you'd use the actual tokenizer.
        For now, we estimate based on word position.
        """
        # Simple word-based tokenization (placeholder)
        # In real implementation, use the model's actual tokenizer
        words = text.lower().split()
        token_groups = {}

        for target in target_tokens:
            target_lower = target.lower()
            indices = []

            for i, word in enumerate(words):
                if target_lower in word:
                    # Add 1 for [CLS] token at start
                    indices.append(i + 1)

            if indices:
                token_groups[target] = indices
            else:
                logger.warning(f"[KSamplerLayered] Token '{target}' not found in prompt")

        return token_groups

    def _setup_progress_tracking(self, model, redis_url: str, job_id: str, steps: int):
        """Setup Redis progress tracking if enabled"""
        if not redis_url or not job_id:
            return model

        try:
            import redis
            redis_client = redis.from_url(redis_url)
            redis_key = f"comfyui:progress:{job_id}"

            # Initialize progress
            progress_data = {
                "step_current": 0,
                "step_total": steps,
                "progress": 0.0,
                "status": "started",
            }
            redis_client.set(redis_key, json.dumps(progress_data), ex=3600)

            # Clone model and add callback
            m = model.clone()
            manager = self

            def progress_callback(args):
                step = getattr(manager, '_current_step', 0) + 1
                manager._current_step = step

                # Update attention manager step
                if manager.attention_manager:
                    manager.attention_manager.set_step(step)

                # Update Redis
                try:
                    progress = step / steps
                    status = "completed" if progress >= 1.0 else "running"
                    progress_data = {
                        "step_current": step,
                        "step_total": steps,
                        "progress": round(progress, 4),
                        "status": status,
                    }
                    redis_client.set(redis_key, json.dumps(progress_data), ex=3600)
                except Exception as e:
                    logger.warning(f"[KSamplerLayered] Progress update failed: {e}")

                return args["denoised"]

            m.set_model_sampler_post_cfg_function(progress_callback)
            logger.info(f"[KSamplerLayered] Progress tracking enabled: {redis_key}")
            return m

        except ImportError:
            logger.warning("[KSamplerLayered] redis package not installed")
            return model
        except Exception as e:
            logger.warning(f"[KSamplerLayered] Progress setup failed: {e}")
            return model

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        extract_tokens="",
        attention_layers="15",
        redis_url="",
        job_id=""
    ):
        """
        Perform sampling with attention capture.
        """
        self._current_step = 0

        # Parse inputs
        target_layers = self._parse_attention_layers(attention_layers)
        token_names = self._parse_extract_tokens(extract_tokens)

        # Setup attention manager
        attention_maps = {}
        if token_names:
            self.attention_manager = AttentionHookManager(
                target_layers=target_layers,
                target_steps=None,  # Capture all steps for now
            )

            # Try to register hooks
            hooks_registered = self.attention_manager.register_hooks(model)
            if not hooks_registered:
                logger.warning("[KSamplerLayered] Could not register attention hooks")
                self.attention_manager = None

        # Setup progress tracking
        working_model = self._setup_progress_tracking(model, redis_url, job_id, steps)

        try:
            # Get latent
            latent = latent_image.copy()
            latent_samples = latent["samples"]

            # Prepare noise
            batch_size = latent_samples.shape[0]
            noise = comfy.sample.prepare_noise(latent_samples, seed, None)

            # Get sampler and sigmas
            sampler = comfy.samplers.KSampler(
                working_model, steps=steps, device=comfy.model_management.get_torch_device(),
                sampler=sampler_name, scheduler=scheduler, denoise=denoise
            )

            # Sample
            samples = sampler.sample(
                noise, positive, negative, cfg=cfg,
                latent_image=latent_samples,
                start_step=0, last_step=steps,
                force_full_denoise=True, denoise_mask=None
            )

            # Build output latent
            out_latent = latent.copy()
            out_latent["samples"] = samples

            # Extract attention maps if enabled
            if self.attention_manager and token_names:
                # Get prompt text from conditioning
                prompt_text = ""
                if positive and len(positive) > 0:
                    cond = positive[0]
                    if isinstance(cond, (list, tuple)) and len(cond) > 1:
                        pooled = cond[1]
                        if isinstance(pooled, dict) and "prompt" in pooled:
                            prompt_text = pooled["prompt"]

                # Find token indices
                token_groups = self._find_token_indices(prompt_text, token_names)

                if token_groups:
                    # Get image size from latent
                    _, _, h, w = latent_samples.shape
                    image_size = (h * 8, w * 8)  # VAE downscale factor

                    # Use middle target layer for heatmaps
                    use_layer = target_layers[len(target_layers) // 2]

                    # Extract heatmaps
                    heatmaps = self.attention_manager.get_all_token_heatmaps(
                        token_groups=token_groups,
                        layer_idx=use_layer,
                        step=-1,  # Last step
                        image_size=image_size
                    )

                    attention_maps = {
                        "heatmaps": heatmaps,
                        "token_groups": token_groups,
                        "layer_idx": use_layer,
                        "image_size": image_size
                    }

                    logger.info(f"[KSamplerLayered] Extracted {len(heatmaps)} attention maps")

        finally:
            # Cleanup hooks
            if self.attention_manager:
                self.attention_manager.remove_hooks()
                self.attention_manager = None

        return (out_latent, attention_maps)


# =============================================================================
# AttentionToMask Node
# =============================================================================

class AttentionToMask:
    """
    Convert attention heatmaps to binary masks.

    Takes attention_maps from KSamplerLayered and produces masks
    that can be used for layer separation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_maps": ("ATTENTION_MAPS",),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "blur_radius": ("INT", {
                    "default": 5, "min": 0, "max": 50, "step": 1
                }),
                "expand_mask": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1
                }),
            },
            "optional": {
                "use_otsu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK_LIST")
    RETURN_NAMES = ("combined_mask", "individual_masks")
    FUNCTION = "convert"
    CATEGORY = "IXIWORKS/Layer"

    def _otsu_threshold(self, heatmap: torch.Tensor) -> float:
        """Compute Otsu's threshold"""
        try:
            from skimage.filters import threshold_otsu
            return threshold_otsu(heatmap.numpy())
        except ImportError:
            # Fallback: simple histogram-based threshold
            hist, bins = np.histogram(heatmap.numpy().flatten(), bins=256)
            total = heatmap.numel()

            sum_total = sum(i * hist[i] for i in range(256))
            sum_bg, weight_bg, max_var, threshold = 0, 0, 0, 0

            for i in range(256):
                weight_bg += hist[i]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break

                sum_bg += i * hist[i]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg

                var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if var > max_var:
                    max_var = var
                    threshold = i

            return threshold / 255.0

    def _apply_morphology(self, mask: torch.Tensor, blur_radius: int, expand: int) -> torch.Tensor:
        """Apply blur and morphological operations"""
        if blur_radius > 0:
            # Gaussian blur
            kernel_size = blur_radius * 2 + 1
            mask = mask.unsqueeze(0).unsqueeze(0).float()
            mask = F.avg_pool2d(
                F.pad(mask, (blur_radius, blur_radius, blur_radius, blur_radius), mode='reflect'),
                kernel_size, stride=1
            ).squeeze()

        if expand != 0:
            # Dilate (positive) or erode (negative)
            kernel_size = abs(expand) * 2 + 1
            padding = abs(expand)

            mask = mask.unsqueeze(0).unsqueeze(0).float()
            if expand > 0:
                mask = F.max_pool2d(
                    F.pad(mask, (padding, padding, padding, padding), mode='constant', value=0),
                    kernel_size, stride=1
                )
            else:
                mask = -F.max_pool2d(
                    F.pad(-mask, (padding, padding, padding, padding), mode='constant', value=-1),
                    kernel_size, stride=1
                )
            mask = mask.squeeze()

        return mask

    def convert(
        self,
        attention_maps: Dict[str, Any],
        threshold: float,
        blur_radius: int,
        expand_mask: int,
        use_otsu: bool = True
    ):
        """Convert attention heatmaps to masks"""
        if not attention_maps or "heatmaps" not in attention_maps:
            # Return empty masks
            empty_mask = torch.zeros((1, 64, 64))
            return (empty_mask, [])

        heatmaps = attention_maps["heatmaps"]
        masks = []

        for name, heatmap in heatmaps.items():
            # Determine threshold
            if use_otsu:
                thresh = self._otsu_threshold(heatmap)
            else:
                thresh = threshold

            # Create binary mask
            mask = (heatmap > thresh).float()

            # Apply morphological operations
            mask = self._apply_morphology(mask, blur_radius, expand_mask)

            # Ensure [0, 1] range
            mask = torch.clamp(mask, 0, 1)

            masks.append({
                "name": name,
                "mask": mask,
                "threshold_used": thresh
            })

            logger.info(f"[AttentionToMask] Created mask for '{name}', threshold={thresh:.3f}")

        # Create combined mask (union of all masks)
        if masks:
            combined = torch.zeros_like(masks[0]["mask"])
            for m in masks:
                combined = torch.maximum(combined, m["mask"])
            combined = combined.unsqueeze(0)  # Add batch dim
        else:
            combined = torch.zeros((1, 64, 64))

        # Format individual masks for output
        mask_list = [(m["name"], m["mask"].unsqueeze(0)) for m in masks]

        return (combined, mask_list)


# =============================================================================
# LayerSeparator Node
# =============================================================================

class LayerSeparator:
    """
    Separate image into layers based on masks.

    Takes an image and masks, outputs individual layer images with alpha.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_list": ("MASK_LIST",),
            },
            "optional": {
                "include_background": ("BOOLEAN", {"default": True}),
                "feather_edges": ("INT", {"default": 3, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES = ("LAYER_LIST",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "separate"
    CATEGORY = "IXIWORKS/Layer"

    def _feather_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply feathering to mask edges"""
        if radius <= 0:
            return mask

        # Gaussian-like feathering using multiple blur passes
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        for _ in range(radius):
            mask = F.avg_pool2d(
                F.pad(mask, (1, 1, 1, 1), mode='reflect'),
                3, stride=1
            )
        return mask.squeeze()

    def separate(
        self,
        image: torch.Tensor,
        mask_list: List[Tuple[str, torch.Tensor]],
        include_background: bool = True,
        feather_edges: int = 3
    ):
        """Separate image into layers"""
        # image shape: (batch, H, W, C) - ComfyUI format
        if len(image.shape) == 4:
            img = image[0]  # Take first batch
        else:
            img = image

        H, W, C = img.shape
        layers = []

        # Track which pixels are used
        used_mask = torch.zeros((H, W), device=img.device)

        for name, mask in mask_list:
            # Ensure mask is right size
            if mask.shape[-2:] != (H, W):
                mask = F.interpolate(
                    mask.unsqueeze(0).float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            if len(mask.shape) == 3:
                mask = mask[0]  # Remove batch dim

            # Feather edges
            mask = self._feather_mask(mask, feather_edges)

            # Create layer with alpha
            layer_rgb = img * mask.unsqueeze(-1)
            layer_alpha = mask

            # Combine into RGBA
            layer_rgba = torch.cat([layer_rgb, layer_alpha.unsqueeze(-1)], dim=-1)

            layers.append({
                "name": name,
                "image": layer_rgba,
                "bounds": self._get_bounds(mask)
            })

            # Update used mask
            used_mask = torch.maximum(used_mask, mask)

            logger.info(f"[LayerSeparator] Created layer '{name}'")

        # Create background layer
        if include_background:
            bg_mask = 1.0 - used_mask
            bg_mask = torch.clamp(bg_mask, 0, 1)

            bg_rgb = img * bg_mask.unsqueeze(-1)
            bg_alpha = torch.ones((H, W), device=img.device)  # Full opacity for background

            bg_rgba = torch.cat([bg_rgb, bg_alpha.unsqueeze(-1)], dim=-1)

            layers.append({
                "name": "background",
                "image": bg_rgba,
                "bounds": {"x": 0, "y": 0, "width": W, "height": H}
            })

            logger.info(f"[LayerSeparator] Created background layer")

        return (layers,)

    def _get_bounds(self, mask: torch.Tensor) -> Dict[str, int]:
        """Get bounding box of mask"""
        nonzero = torch.nonzero(mask > 0.5)
        if len(nonzero) == 0:
            return {"x": 0, "y": 0, "width": 0, "height": 0}

        y_min, x_min = nonzero.min(dim=0)[0].tolist()
        y_max, x_max = nonzero.max(dim=0)[0].tolist()

        return {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min + 1),
            "height": int(y_max - y_min + 1)
        }


# =============================================================================
# LayerExporter Node
# =============================================================================

class LayerExporter:
    """
    Export layers to files (PNG + JSON or PSD).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("LAYER_LIST",),
                "output_dir": ("STRING", {"default": "output/layers"}),
                "filename_prefix": ("STRING", {"default": "scene"}),
            },
            "optional": {
                "export_psd": ("BOOLEAN", {"default": False}),
                "export_png": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "export"
    CATEGORY = "IXIWORKS/Layer"
    OUTPUT_NODE = True

    def export(
        self,
        layers: List[Dict],
        output_dir: str,
        filename_prefix: str,
        export_psd: bool = False,
        export_png: bool = True
    ):
        """Export layers to files"""
        import os

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        composition = {
            "version": "1.0",
            "layers": []
        }

        for i, layer in enumerate(layers):
            name = layer["name"]
            img_tensor = layer["image"]  # (H, W, 4) RGBA
            bounds = layer["bounds"]

            # Convert to numpy and then PIL
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGBA')

            if export_png:
                # Save PNG
                png_filename = f"{filename_prefix}_{name}.png"
                png_path = os.path.join(output_dir, png_filename)
                img_pil.save(png_path, 'PNG')
                logger.info(f"[LayerExporter] Saved {png_path}")

                composition["layers"].append({
                    "name": name,
                    "file": png_filename,
                    "bounds": bounds,
                    "order": i
                })

        # Save composition JSON
        json_path = os.path.join(output_dir, f"{filename_prefix}_composition.json")
        with open(json_path, 'w') as f:
            json.dump(composition, f, indent=2)
        logger.info(f"[LayerExporter] Saved {json_path}")

        # Export PSD if requested
        if export_psd:
            try:
                self._export_psd(layers, output_dir, filename_prefix)
            except Exception as e:
                logger.warning(f"[LayerExporter] PSD export failed: {e}")

        return (output_dir,)

    def _export_psd(self, layers: List[Dict], output_dir: str, filename_prefix: str):
        """Export to PSD format"""
        try:
            from psd_tools import PSDImage
            from psd_tools.api.layers import PixelLayer
        except ImportError:
            logger.warning("[LayerExporter] psd-tools not installed, skipping PSD export")
            return

        # PSD export implementation would go here
        # This is a placeholder - psd-tools has limited write support
        logger.info("[LayerExporter] PSD export not yet implemented")


# =============================================================================
# ModelStructureDebug Node
# =============================================================================

class ModelStructureDebug:
    """
    Debug node to inspect model structure for attention hook compatibility.

    Outputs detailed information about the model's internal structure,
    helping identify the correct path to attention layers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "max_depth": ("INT", {"default": 4, "min": 1, "max": 10}),
                "show_shapes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "structure_info")
    FUNCTION = "debug"
    CATEGORY = "IXIWORKS/Debug"
    OUTPUT_NODE = True

    def _explore_module(self, module, prefix="", depth=0, max_depth=4, show_shapes=True):
        """Recursively explore module structure"""
        lines = []

        if depth > max_depth:
            return lines

        indent = "  " * depth

        # Module type
        module_type = type(module).__name__
        lines.append(f"{indent}{prefix}: {module_type}")

        # Check for important attributes
        important_attrs = ['blocks', 'transformer_blocks', 'layers', 'attn', 'self_attn',
                          'attention', 'to_q', 'to_k', 'to_v', 'num_heads', 'head_dim']

        for attr in important_attrs:
            if hasattr(module, attr):
                obj = getattr(module, attr)
                if isinstance(obj, (int, float, str, bool)):
                    lines.append(f"{indent}  .{attr} = {obj}")
                elif hasattr(obj, '__len__') and not isinstance(obj, str):
                    lines.append(f"{indent}  .{attr} = {type(obj).__name__}[{len(obj)}]")
                elif hasattr(obj, 'weight') and show_shapes:
                    shape = tuple(obj.weight.shape)
                    lines.append(f"{indent}  .{attr} = Linear{shape}")
                else:
                    lines.append(f"{indent}  .{attr} = {type(obj).__name__}")

        # Recursively explore children
        if hasattr(module, 'named_children'):
            for name, child in module.named_children():
                if name in ['blocks', 'transformer_blocks', 'layers']:
                    # For block lists, only show first few
                    if hasattr(child, '__len__') and len(child) > 0:
                        lines.append(f"{indent}  .{name} = ModuleList[{len(child)}]")
                        # Show first block in detail
                        lines.append(f"{indent}    [0] (first block):")
                        lines.extend(self._explore_module(
                            child[0], prefix="", depth=depth+3,
                            max_depth=max_depth, show_shapes=show_shapes
                        ))
                        if len(child) > 1:
                            lines.append(f"{indent}    ... ({len(child)-1} more blocks)")
                elif depth < max_depth - 1:
                    lines.extend(self._explore_module(
                        child, prefix=f".{name}", depth=depth+1,
                        max_depth=max_depth, show_shapes=show_shapes
                    ))

        return lines

    def _find_attention_path(self, model):
        """Try to find the path to attention layers"""
        paths_to_try = [
            # ComfyUI typical paths
            ('model', 'diffusion_model', 'blocks'),
            ('model', 'diffusion_model', 'transformer_blocks'),
            ('model', 'diffusion_model', 'layers'),
            # Direct paths
            ('diffusion_model', 'blocks'),
            ('diffusion_model', 'transformer_blocks'),
            # Other possibilities
            ('blocks',),
            ('transformer_blocks',),
            ('layers',),
        ]

        results = []

        for path in paths_to_try:
            obj = model
            valid = True
            full_path = "model"

            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                    full_path += f".{attr}"
                else:
                    valid = False
                    break

            if valid and hasattr(obj, '__len__') and len(obj) > 0:
                # Found blocks, check for attention
                block = obj[0]
                attn_attr = None

                for attn_name in ['attn', 'self_attn', 'attention', 'self_attention']:
                    if hasattr(block, attn_name):
                        attn_attr = attn_name
                        break

                if attn_attr:
                    attn_module = getattr(block, attn_attr)
                    has_to_q = hasattr(attn_module, 'to_q')
                    has_to_k = hasattr(attn_module, 'to_k')
                    num_heads = getattr(attn_module, 'num_heads', 'unknown')

                    # Collect all sub-modules and attributes of attention
                    attn_children = {}
                    if hasattr(attn_module, 'named_children'):
                        for name, child in attn_module.named_children():
                            child_info = type(child).__name__
                            if hasattr(child, 'weight'):
                                child_info += f" weight={tuple(child.weight.shape)}"
                            attn_children[name] = child_info

                    # Also check common Q, K attribute names
                    qk_candidates = ['to_q', 'to_k', 'to_v', 'q_proj', 'k_proj', 'v_proj',
                                     'wq', 'wk', 'wv', 'query', 'key', 'value',
                                     'q', 'k', 'v', 'qkv', 'in_proj', 'qkv_proj']
                    found_qk = {name: True for name in qk_candidates if hasattr(attn_module, name)}

                    # Collect block-level children too
                    block_children = {}
                    if hasattr(block, 'named_children'):
                        for name, child in block.named_children():
                            child_info = type(child).__name__
                            if hasattr(child, 'weight'):
                                child_info += f" weight={tuple(child.weight.shape)}"
                            elif hasattr(child, '__len__'):
                                child_info += f" [{len(child)}]"
                            block_children[name] = child_info

                    results.append({
                        'path': full_path,
                        'num_blocks': len(obj),
                        'attn_attr': attn_attr,
                        'has_to_q': has_to_q,
                        'has_to_k': has_to_k,
                        'num_heads': num_heads,
                        'hook_compatible': has_to_q and has_to_k,
                        'attn_type': type(attn_module).__name__,
                        'attn_children': attn_children,
                        'found_qk_attrs': found_qk,
                        'block_children': block_children,
                    })

        return results

    def debug(self, model, max_depth=4, show_shapes=True):
        """Debug model structure"""
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("MODEL STRUCTURE DEBUG")
        output_lines.append("=" * 60)

        # Basic info
        output_lines.append(f"\nModel type: {type(model).__name__}")

        # Explore structure
        output_lines.append("\n" + "-" * 60)
        output_lines.append("STRUCTURE TREE")
        output_lines.append("-" * 60)

        structure_lines = self._explore_module(
            model, prefix="model", depth=0,
            max_depth=max_depth, show_shapes=show_shapes
        )
        output_lines.extend(structure_lines)

        # Find attention paths
        output_lines.append("\n" + "-" * 60)
        output_lines.append("ATTENTION PATH ANALYSIS")
        output_lines.append("-" * 60)

        attention_paths = self._find_attention_path(model)

        if attention_paths:
            for i, path_info in enumerate(attention_paths):
                output_lines.append(f"\n[Path {i+1}]")
                output_lines.append(f"  Blocks: {path_info['path']}")
                output_lines.append(f"  Num blocks: {path_info['num_blocks']}")
                output_lines.append(f"  Attention attr: .{path_info['attn_attr']}")
                output_lines.append(f"  Attention type: {path_info.get('attn_type', 'unknown')}")
                output_lines.append(f"  has to_q: {path_info['has_to_q']}")
                output_lines.append(f"  has to_k: {path_info['has_to_k']}")
                output_lines.append(f"  num_heads: {path_info['num_heads']}")
                output_lines.append(f"  Hook compatible: {'YES ✓' if path_info['hook_compatible'] else 'NO ✗'}")

                # Show Q, K attribute candidates found
                found_qk = path_info.get('found_qk_attrs', {})
                if found_qk:
                    output_lines.append(f"\n  Found Q/K/V attributes: {list(found_qk.keys())}")
                else:
                    output_lines.append(f"\n  No standard Q/K/V attributes found")

                # Show all attention sub-modules
                attn_children = path_info.get('attn_children', {})
                if attn_children:
                    output_lines.append(f"\n  Attention sub-modules:")
                    for name, info in attn_children.items():
                        output_lines.append(f"    .{name}: {info}")
                else:
                    output_lines.append(f"\n  No attention sub-modules found")

                # Show block-level children
                block_children = path_info.get('block_children', {})
                if block_children:
                    output_lines.append(f"\n  Block sub-modules:")
                    for name, info in block_children.items():
                        output_lines.append(f"    .{name}: {info}")
        else:
            output_lines.append("\n[WARNING] Could not find attention layers!")
            output_lines.append("The model structure may be different from expected.")
            output_lines.append("Check the structure tree above for clues.")

        # Recommendations
        output_lines.append("\n" + "-" * 60)
        output_lines.append("RECOMMENDATIONS")
        output_lines.append("-" * 60)

        if attention_paths:
            compatible = [p for p in attention_paths if p['hook_compatible']]
            if compatible:
                best = compatible[0]
                output_lines.append(f"\n✓ Hook registration should work!")
                output_lines.append(f"  Use path: {best['path']}[i].{best['attn_attr']}")
                output_lines.append(f"  Recommended layers: [10, 15, 20] (of {best['num_blocks']})")
            else:
                output_lines.append("\n✗ No compatible attention modules found")
                output_lines.append("  May need to modify layer_nodes.py")
        else:
            output_lines.append("\n✗ Could not analyze model")
            output_lines.append("  Please share this output for debugging")

        output_lines.append("\n" + "=" * 60)

        # Join all lines
        structure_info = "\n".join(output_lines)

        # Print to console
        print(structure_info)

        # Also log
        logger.info(f"[ModelStructureDebug] Analysis complete")

        return (model, structure_info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "KSamplerLayered": KSamplerLayered,
    "AttentionToMask": AttentionToMask,
    "LayerSeparator": LayerSeparator,
    "LayerExporter": LayerExporter,
    "ModelStructureDebug": ModelStructureDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerLayered": "KSampler Layered (Attention Extract)",
    "AttentionToMask": "Attention To Mask",
    "LayerSeparator": "Layer Separator",
    "LayerExporter": "Layer Exporter",
    "ModelStructureDebug": "Model Structure Debug",
}
