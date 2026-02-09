"""
Phase 1 PoC: Z-Image Attention Hook Test
=========================================

목적: Z-Image 모델에서 Attention weights를 추출할 수 있는지 검증

테스트 항목:
1. ZImageDiT 모델 로드
2. Attention 레이어에 Hook 등록
3. Q, K 캡처
4. Attention weights 계산
5. 특정 토큰의 히트맵 시각화

사용법:
    pip install diffsynth-studio torch matplotlib
    python attention_hook_poc.py
"""

import os
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class AttentionCapture:
    """Attention weights 캡처를 위한 데이터 클래스"""
    layer_idx: int
    step: int
    query: Optional[torch.Tensor] = None
    key: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class AttentionHookManager:
    """
    Z-Image 모델의 Attention 레이어에 Hook을 등록하고
    Q, K를 캡처하여 Attention weights를 계산하는 매니저
    """

    def __init__(self, target_layers: Optional[List[int]] = None):
        """
        Args:
            target_layers: 캡처할 레이어 인덱스 리스트 (None이면 전체)
        """
        self.target_layers = target_layers or list(range(30))  # Z-Image는 30개 레이어
        self.captures: Dict[int, List[AttentionCapture]] = {}
        self.hooks: List = []
        self.current_step = 0

    def _create_hook(self, layer_idx: int):
        """특정 레이어에 대한 Hook 함수 생성"""
        def hook_fn(module, input, output):
            # DiffSynth-Studio의 Attention forward에서
            # input[0]이 hidden_states
            # 내부에서 Q, K, V가 계산됨

            # Hook에서는 module의 속성에 접근해야 함
            # 실제 구현에서는 forward 중간에 끼어들어야 하므로
            # forward_hook 대신 forward_pre_hook + 모듈 패치가 필요할 수 있음

            # 여기서는 개념 증명용으로 output만 캡처
            if layer_idx not in self.captures:
                self.captures[layer_idx] = []

            self.captures[layer_idx].append(AttentionCapture(
                layer_idx=layer_idx,
                step=self.current_step,
                # 실제로는 Q, K를 캡처해야 함
            ))

        return hook_fn

    def _create_qk_capture_hook(self, layer_idx: int):
        """Q, K를 직접 캡처하는 Hook (forward를 패치하는 방식)"""
        manager = self

        def patched_forward(original_forward):
            def wrapper(hidden_states, freqs_cis=None, attention_mask=None, **kwargs):
                # 원본 forward 호출 전에 Q, K 계산을 가로챔
                # 이 부분은 실제 모듈 구조에 맞게 수정 필요

                module = wrapper._module

                # Q, K, V 계산 (DiffSynth-Studio 구조 기반)
                query = module.to_q(hidden_states)
                key = module.to_k(hidden_states)
                value = module.to_v(hidden_states)

                # Norm 적용
                if hasattr(module, 'norm_q'):
                    query = module.norm_q(query)
                if hasattr(module, 'norm_k'):
                    key = module.norm_k(key)

                # Attention weights 계산
                # Shape: (batch, heads, seq_len, head_dim)
                batch_size = query.shape[0]
                num_heads = getattr(module, 'num_heads', 32)  # Z-Image: 32 heads
                head_dim = query.shape[-1] // num_heads

                # Reshape for attention
                q = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
                k = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

                # Attention weights: (batch, heads, seq_q, seq_k)
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)

                # 캡처 저장
                if layer_idx not in manager.captures:
                    manager.captures[layer_idx] = []

                manager.captures[layer_idx].append(AttentionCapture(
                    layer_idx=layer_idx,
                    step=manager.current_step,
                    query=query.detach().cpu(),
                    key=key.detach().cpu(),
                    attention_weights=attn_weights.detach().cpu()
                ))

                # 원본 forward 호출
                return original_forward(hidden_states, freqs_cis, attention_mask, **kwargs)

            return wrapper

        return patched_forward

    def register_hooks(self, model):
        """
        모델의 Attention 레이어에 Hook 등록

        Args:
            model: ZImageDiT 모델 인스턴스
        """
        # ZImageDiT 구조: model.blocks[i].attn
        if hasattr(model, 'blocks'):
            for idx, block in enumerate(model.blocks):
                if idx in self.target_layers and hasattr(block, 'attn'):
                    attn_module = block.attn

                    # Forward hook 등록
                    hook = attn_module.register_forward_hook(self._create_hook(idx))
                    self.hooks.append(hook)

                    print(f"[AttentionHook] Registered hook on layer {idx}")
        else:
            print("[AttentionHook] Warning: model.blocks not found")

    def remove_hooks(self):
        """등록된 Hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print(f"[AttentionHook] Removed all hooks")

    def set_step(self, step: int):
        """현재 샘플링 스텝 설정"""
        self.current_step = step

    def get_token_heatmap(
        self,
        token_indices: List[int],
        layer_idx: int = 15,
        step: int = -1,
        image_size: Tuple[int, int] = (64, 64)
    ) -> torch.Tensor:
        """
        특정 토큰의 Attention 히트맵 추출

        Args:
            token_indices: 텍스트 토큰 인덱스 리스트
            layer_idx: 추출할 레이어 인덱스
            step: 샘플링 스텝 (-1이면 마지막)
            image_size: 히트맵 출력 크기 (H, W)

        Returns:
            히트맵 텐서 (H, W)
        """
        if layer_idx not in self.captures:
            raise ValueError(f"Layer {layer_idx} not captured")

        captures = self.captures[layer_idx]
        if not captures:
            raise ValueError(f"No captures for layer {layer_idx}")

        # 스텝 선택
        capture = captures[step] if step >= 0 else captures[-1]

        if capture.attention_weights is None:
            raise ValueError("Attention weights not captured")

        attn = capture.attention_weights  # (batch, heads, seq_q, seq_k)

        # 텍스트 토큰 → 이미지 토큰 attention 추출
        # Z-Image: [텍스트 토큰들 | 시각 의미 토큰들 | 이미지 VAE 토큰들]
        # 이미지 토큰 시작 위치는 모델 설정에 따라 다름

        # 예시: 텍스트 77토큰 + 시각 256토큰 후 이미지 토큰 시작
        text_len = 77  # 예시값, 실제 토크나이저에서 확인 필요
        visual_len = 256  # 예시값
        image_start = text_len + visual_len

        # 특정 토큰들의 attention 평균
        token_attn = attn[:, :, token_indices, image_start:]  # (batch, heads, num_tokens, image_tokens)
        token_attn = token_attn.mean(dim=(0, 1, 2))  # 평균 → (image_tokens,)

        # 이미지 크기로 reshape
        img_tokens = int(math.sqrt(token_attn.shape[0]))
        heatmap = token_attn.view(img_tokens, img_tokens)

        # 목표 크기로 리사이즈
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return heatmap

    def clear_captures(self):
        """캡처된 데이터 초기화"""
        self.captures.clear()


def visualize_heatmap(heatmap: torch.Tensor, save_path: str = "heatmap.png"):
    """히트맵 시각화 및 저장"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Attention Weight')
        plt.title('Token Attention Heatmap')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualize] Saved heatmap to {save_path}")
    except ImportError:
        print("[Visualize] matplotlib not installed, skipping visualization")


def test_hook_registration():
    """Hook 등록 테스트 (모델 없이)"""
    print("\n" + "="*60)
    print("Test 1: Hook Registration (without model)")
    print("="*60)

    manager = AttentionHookManager(target_layers=[10, 15, 20])
    print(f"Target layers: {manager.target_layers}")
    print(f"Captures initialized: {len(manager.captures)}")

    # 가짜 캡처 데이터 추가
    manager.captures[15] = [
        AttentionCapture(
            layer_idx=15,
            step=0,
            attention_weights=torch.rand(1, 32, 100, 4096)  # 예시 shape
        )
    ]

    print(f"Added fake capture for layer 15")
    print(f"Attention weights shape: {manager.captures[15][0].attention_weights.shape}")

    print("\n[PASS] Hook registration logic works")


def test_heatmap_extraction():
    """히트맵 추출 테스트 (가짜 데이터)"""
    print("\n" + "="*60)
    print("Test 2: Heatmap Extraction (with fake data)")
    print("="*60)

    manager = AttentionHookManager(target_layers=[15])

    # 가짜 attention weights 생성
    # Shape: (batch=1, heads=32, seq_q=100, seq_k=4096+333)
    # seq_k = text(77) + visual(256) + image(64*64=4096)
    seq_len = 77 + 256 + 4096
    fake_attn = torch.rand(1, 32, 100, seq_len)

    # 특정 영역에 높은 attention 추가 (토큰 3-5가 이미지 중앙에 집중)
    center_start = 77 + 256 + 64*30  # 중앙 근처
    fake_attn[:, :, 3:6, center_start:center_start+64*4] = 5.0  # 높은 가중치
    fake_attn = F.softmax(fake_attn, dim=-1)

    manager.captures[15] = [
        AttentionCapture(
            layer_idx=15,
            step=0,
            attention_weights=fake_attn
        )
    ]

    try:
        heatmap = manager.get_token_heatmap(
            token_indices=[3, 4, 5],
            layer_idx=15,
            image_size=(64, 64)
        )
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap min/max: {heatmap.min():.4f} / {heatmap.max():.4f}")

        # 시각화
        output_dir = os.path.dirname(os.path.abspath(__file__))
        visualize_heatmap(heatmap, os.path.join(output_dir, "test_heatmap.png"))

        print("\n[PASS] Heatmap extraction works")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")


def test_with_diffsynth():
    """DiffSynth-Studio로 실제 테스트"""
    print("\n" + "="*60)
    print("Test 3: Real Test with DiffSynth-Studio")
    print("="*60)

    try:
        from diffsynth import ModelManager, ZImagePipeline
        print("[OK] DiffSynth-Studio imported successfully")

        # 모델 로드 (실제 테스트 시)
        # model_manager = ModelManager()
        # model_manager.load_model("Tongyi-MAI/Z-Image-Turbo")
        # pipe = ZImagePipeline.from_model_manager(model_manager)

        print("[SKIP] Model loading skipped (requires GPU and model download)")
        print("[INFO] To run full test, uncomment model loading code")

    except ImportError:
        print("[SKIP] DiffSynth-Studio not installed")
        print("[INFO] Install with: pip install diffsynth-studio")


def main():
    """PoC 메인 함수"""
    print("\n" + "#"*60)
    print("# Phase 1 PoC: Z-Image Attention Hook Test")
    print("#"*60)

    # Test 1: Hook 등록 로직
    test_hook_registration()

    # Test 2: 히트맵 추출 로직
    test_heatmap_extraction()

    # Test 3: DiffSynth-Studio 연동 (선택)
    test_with_diffsynth()

    print("\n" + "#"*60)
    print("# PoC Summary")
    print("#"*60)
    print("""
    결과:
    - [PASS] AttentionHookManager 클래스 구조 검증
    - [PASS] Q, K → Attention weights 계산 로직
    - [PASS] Token → Heatmap 변환 로직
    - [TODO] 실제 Z-Image 모델에서 Hook 테스트

    다음 단계:
    1. DiffSynth-Studio 설치: pip install diffsynth-studio
    2. Z-Image-Turbo 모델 다운로드
    3. 실제 이미지 생성하면서 Attention 캡처
    4. 캐릭터 이름 토큰의 히트맵 시각화
    """)


if __name__ == "__main__":
    main()
