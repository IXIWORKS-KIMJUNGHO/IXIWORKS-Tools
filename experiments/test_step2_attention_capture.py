"""
Step 2: Attention 캡처 테스트
=============================

목적: 실제 이미지 생성하면서 Attention weights 캡처

실행:
    python test_step2_attention_capture.py

필요 환경:
    - GPU 서버 (24GB+ VRAM)
    - diffsynth-studio 설치
    - Step 1 통과
"""

import os
import sys
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# Attention Hook Manager (from layer_nodes.py)
# =============================================================================

class AttentionHookManager:
    """Attention 캡처용 Hook 매니저"""

    def __init__(self, target_layers: List[int] = None):
        self.target_layers = target_layers or [10, 15, 20]
        self.captures = {}
        self.hooks = []
        self.current_step = 0
        self._enabled = True

    def _create_hook(self, layer_idx: int):
        manager = self

        def hook_fn(module, input, output):
            if not manager._enabled:
                return

            try:
                hidden_states = input[0]

                if not hasattr(module, 'to_q') or not hasattr(module, 'to_k'):
                    return

                query = module.to_q(hidden_states)
                key = module.to_k(hidden_states)

                if hasattr(module, 'norm_q'):
                    query = module.norm_q(query)
                if hasattr(module, 'norm_k'):
                    key = module.norm_k(key)

                batch_size = query.shape[0]
                seq_len = query.shape[1]
                num_heads = getattr(module, 'num_heads', 32)
                head_dim = query.shape[-1] // num_heads

                q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)

                if layer_idx not in manager.captures:
                    manager.captures[layer_idx] = []

                manager.captures[layer_idx].append({
                    "step": manager.current_step,
                    "attention_weights": attn_weights.detach().cpu(),
                    "seq_len": seq_len,
                })

                print(f"  [Hook] Layer {layer_idx}, Step {manager.current_step}, "
                      f"Attn shape: {tuple(attn_weights.shape)}")

            except Exception as e:
                print(f"  [Hook Error] Layer {layer_idx}: {e}")

        return hook_fn

    def register_hooks(self, dit_model, attn_attr_name="attn"):
        """DiT 모델에 Hook 등록"""
        blocks = None
        if hasattr(dit_model, 'blocks'):
            blocks = dit_model.blocks
        elif hasattr(dit_model, 'transformer_blocks'):
            blocks = dit_model.transformer_blocks

        if blocks is None:
            print("[ERROR] Could not find transformer blocks")
            return False

        for idx in self.target_layers:
            if idx >= len(blocks):
                continue

            block = blocks[idx]
            attn_module = getattr(block, attn_attr_name, None)

            if attn_module is None:
                continue

            hook = attn_module.register_forward_hook(self._create_hook(idx))
            self.hooks.append(hook)
            print(f"[Hook] Registered on layer {idx}")

        return len(self.hooks) > 0

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def set_step(self, step: int):
        self.current_step = step


# =============================================================================
# Test Function
# =============================================================================

def test_attention_capture():
    """실제 이미지 생성하면서 Attention 캡처"""
    print("=" * 60)
    print("Step 2: Attention Capture Test")
    print("=" * 60)

    try:
        from diffsynth import ModelManager, ZImagePipeline
        print("[OK] diffsynth imported")
    except ImportError:
        print("[ERROR] diffsynth not installed")
        return False

    # 모델 로드
    print("\n[INFO] Loading Z-Image-Turbo model...")
    model_manager = ModelManager()

    try:
        model_manager.load_models(["Tongyi-MAI/Z-Image-Turbo"])
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False

    # Pipeline 생성
    print("\n[INFO] Creating pipeline...")
    try:
        pipe = ZImagePipeline.from_model_manager(model_manager)
        print("[OK] Pipeline created")
    except Exception as e:
        print(f"[ERROR] Pipeline creation failed: {e}")
        return False

    # DiT 모델 찾기
    print("\n[INFO] Finding DiT model...")
    dit_model = None

    # 여러 경로 시도
    if hasattr(pipe, 'dit'):
        dit_model = pipe.dit
    elif hasattr(pipe, 'transformer'):
        dit_model = pipe.transformer
    elif hasattr(model_manager, 'dit'):
        dit_model = model_manager.dit

    if dit_model is None:
        print("[ERROR] Could not find DiT model")
        print("Available pipe attributes:", [a for a in dir(pipe) if not a.startswith('_')])
        return False

    print(f"[OK] Found DiT model: {type(dit_model).__name__}")

    # Attention Hook 등록
    print("\n[INFO] Registering attention hooks...")
    hook_manager = AttentionHookManager(target_layers=[10, 15, 20])

    # attn 모듈 이름 확인
    attn_name = "attn"
    if hasattr(dit_model, 'blocks') and len(dit_model.blocks) > 0:
        block = dit_model.blocks[0]
        for name in ['attn', 'self_attn', 'attention']:
            if hasattr(block, name):
                attn_name = name
                break

    success = hook_manager.register_hooks(dit_model, attn_attr_name=attn_name)
    if not success:
        print("[ERROR] Failed to register hooks")
        return False

    # 이미지 생성
    print("\n[INFO] Generating test image...")
    print("-" * 60)

    prompt = "A warrior named Mina standing in a forest, ink sketch style"

    try:
        # Step callback으로 step 추적
        def step_callback(step, timestep, latents):
            hook_manager.set_step(step)
            print(f"\n[Step {step}]")
            return latents

        # 이미지 생성 (적은 step으로 빠르게 테스트)
        image = pipe(
            prompt=prompt,
            num_inference_steps=8,  # 빠른 테스트용
            height=512,
            width=512,
            guidance_scale=5.0,
            callback=step_callback,
            callback_steps=1,
        )

        print("\n[OK] Image generated")

    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        hook_manager.remove_hooks()
        return False

    # Hook 제거
    hook_manager.remove_hooks()

    # 결과 분석
    print("\n" + "=" * 60)
    print("Capture Results")
    print("=" * 60)

    for layer_idx, captures in hook_manager.captures.items():
        print(f"\nLayer {layer_idx}: {len(captures)} captures")
        if captures:
            first = captures[0]
            attn = first["attention_weights"]
            print(f"  Attention shape: {tuple(attn.shape)}")
            print(f"  Sequence length: {first['seq_len']}")
            print(f"  Min/Max: {attn.min():.4f} / {attn.max():.4f}")

    # 히트맵 추출 테스트
    print("\n" + "=" * 60)
    print("Heatmap Extraction Test")
    print("=" * 60)

    if 15 in hook_manager.captures and hook_manager.captures[15]:
        attn = hook_manager.captures[15][-1]["attention_weights"]  # 마지막 step
        seq_len = hook_manager.captures[15][-1]["seq_len"]

        print(f"Using layer 15, last step")
        print(f"Attention shape: {tuple(attn.shape)}")

        # 토큰 구조 추정
        # Z-Image: text(~77) + visual(~256) + image(64*64 for 512px)
        # 실제 값은 모델에 따라 다를 수 있음
        text_len = 77
        visual_len = 256
        image_tokens = 64 * 64  # 512x512 / 8 / 8

        estimated_total = text_len + visual_len + image_tokens
        print(f"\nEstimated token structure:")
        print(f"  Text: 0-{text_len-1} ({text_len} tokens)")
        print(f"  Visual: {text_len}-{text_len+visual_len-1} ({visual_len} tokens)")
        print(f"  Image: {text_len+visual_len}+ ({image_tokens} tokens)")
        print(f"  Total estimated: {estimated_total}")
        print(f"  Actual seq_len: {seq_len}")

        # "Mina" 토큰 위치 추정 (간단히 5-6번 토큰으로 가정)
        mina_tokens = [5, 6]  # "A warrior named Mina..."에서 대략적 위치
        image_start = text_len + visual_len

        if seq_len >= image_start + image_tokens:
            # text→image attention 추출
            text_to_image = attn[:, :, mina_tokens, image_start:image_start+image_tokens]
            print(f"\nText→Image attention shape: {tuple(text_to_image.shape)}")

            # 평균하여 히트맵 생성
            heatmap = text_to_image.mean(dim=(0, 1, 2))  # (4096,)
            print(f"Heatmap shape (flat): {tuple(heatmap.shape)}")

            # 64x64로 reshape
            heatmap_2d = heatmap.view(64, 64)
            print(f"Heatmap shape (2D): {tuple(heatmap_2d.shape)}")
            print(f"Heatmap min/max: {heatmap_2d.min():.6f} / {heatmap_2d.max():.6f}")

            # 히트맵 저장
            try:
                import matplotlib.pyplot as plt

                output_dir = os.path.dirname(os.path.abspath(__file__))

                # 히트맵 시각화
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(heatmap_2d.numpy(), cmap='hot')
                plt.colorbar()
                plt.title('Raw Attention Heatmap')

                # Normalize
                heatmap_norm = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

                plt.subplot(1, 3, 2)
                plt.imshow(heatmap_norm.numpy(), cmap='hot')
                plt.colorbar()
                plt.title('Normalized Heatmap')

                # 생성된 이미지와 오버레이
                plt.subplot(1, 3, 3)
                if hasattr(image, 'images') and len(image.images) > 0:
                    gen_img = image.images[0]
                elif isinstance(image, list) and len(image) > 0:
                    gen_img = image[0]
                else:
                    gen_img = image

                # 512x512로 히트맵 리사이즈
                heatmap_resized = F.interpolate(
                    heatmap_norm.unsqueeze(0).unsqueeze(0),
                    size=(512, 512),
                    mode='bilinear'
                ).squeeze().numpy()

                plt.imshow(gen_img)
                plt.imshow(heatmap_resized, cmap='hot', alpha=0.5)
                plt.title('Overlay on Generated Image')

                plt.tight_layout()
                output_path = os.path.join(output_dir, "step2_attention_result.png")
                plt.savefig(output_path, dpi=150)
                plt.close()

                print(f"\n[OK] Saved visualization to {output_path}")

                # 생성된 이미지도 저장
                img_path = os.path.join(output_dir, "step2_generated_image.png")
                gen_img.save(img_path)
                print(f"[OK] Saved generated image to {img_path}")

            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}")

        else:
            print(f"\n[WARNING] Sequence length mismatch")
            print(f"  Cannot extract text→image attention")

    # 요약
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_captures = sum(len(c) for c in hook_manager.captures.values())
    print(f"""
Test completed successfully!

Captures: {total_captures} total across {len(hook_manager.captures)} layers
Layers captured: {list(hook_manager.captures.keys())}

If heatmap shows concentration in expected regions:
  → Attention extraction is working correctly
  → Proceed to Step 3: Full pipeline test in ComfyUI

If heatmap is uniform or doesn't match character position:
  → May need to adjust token indices
  → May need to try different layers
  → Check tokenizer output for actual token positions
""")

    return True


def main():
    success = test_attention_capture()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
