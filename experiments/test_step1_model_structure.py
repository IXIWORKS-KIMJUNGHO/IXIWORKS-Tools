"""
Step 1: Z-Image 모델 구조 탐색
================================

목적: Hook을 걸 수 있는 Attention 레이어 위치 확인

실행:
    python test_step1_model_structure.py

필요 환경:
    - GPU 서버 (24GB+ VRAM)
    - diffsynth-studio 설치
"""

import sys


def explore_model_structure():
    """Z-Image 모델 구조 탐색"""
    print("=" * 60)
    print("Step 1: Z-Image Model Structure Exploration")
    print("=" * 60)

    try:
        from diffsynth import ModelManager
        print("[OK] diffsynth imported")
    except ImportError:
        print("[ERROR] diffsynth not installed")
        print("Install: pip install diffsynth-studio")
        return False

    # 모델 로드
    print("\n[INFO] Loading Z-Image-Turbo model...")
    print("[INFO] This may take a while on first run (downloading model)")

    try:
        model_manager = ModelManager()
        model_manager.load_models(["Tongyi-MAI/Z-Image-Turbo"])
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False

    # 모델 구조 탐색
    print("\n" + "-" * 60)
    print("Model Structure Analysis")
    print("-" * 60)

    # DIT 모델 찾기
    dit_model = None
    for name, model in model_manager.model.items():
        print(f"\n[Model] {name}: {type(model).__name__}")

        if "dit" in name.lower() or "transformer" in name.lower():
            dit_model = model
            print(f"  → Found DiT model!")

    if dit_model is None:
        # 다른 방법으로 찾기
        print("\n[INFO] Searching for DiT model in model_manager...")
        if hasattr(model_manager, 'dit'):
            dit_model = model_manager.dit
        elif hasattr(model_manager, 'transformer'):
            dit_model = model_manager.transformer

    if dit_model is None:
        print("[WARNING] Could not find DiT model directly")
        print("[INFO] Listing all attributes...")
        for attr in dir(model_manager):
            if not attr.startswith('_'):
                obj = getattr(model_manager, attr)
                print(f"  {attr}: {type(obj).__name__}")
        return False

    # Transformer 블록 탐색
    print("\n" + "-" * 60)
    print("Transformer Blocks Analysis")
    print("-" * 60)

    blocks = None
    if hasattr(dit_model, 'blocks'):
        blocks = dit_model.blocks
    elif hasattr(dit_model, 'transformer_blocks'):
        blocks = dit_model.transformer_blocks
    elif hasattr(dit_model, 'layers'):
        blocks = dit_model.layers

    if blocks is None:
        print("[WARNING] Could not find transformer blocks")
        print("[INFO] DiT model attributes:")
        for attr in dir(dit_model):
            if not attr.startswith('_'):
                obj = getattr(dit_model, attr)
                if hasattr(obj, '__len__'):
                    print(f"  {attr}: {type(obj).__name__} (len={len(obj)})")
                else:
                    print(f"  {attr}: {type(obj).__name__}")
        return False

    print(f"\n[OK] Found {len(blocks)} transformer blocks")

    # 첫 번째 블록 상세 분석
    print("\n" + "-" * 60)
    print("Block Structure (Block 0)")
    print("-" * 60)

    block = blocks[0]
    print(f"Block type: {type(block).__name__}")

    for attr in dir(block):
        if not attr.startswith('_'):
            obj = getattr(block, attr)
            if hasattr(obj, 'parameters'):
                # nn.Module인 경우
                print(f"  [Module] {attr}: {type(obj).__name__}")

    # Attention 모듈 찾기
    print("\n" + "-" * 60)
    print("Attention Module Analysis")
    print("-" * 60)

    attn_module = None
    attn_name = None
    for name in ['attn', 'self_attn', 'attention', 'self_attention']:
        if hasattr(block, name):
            attn_module = getattr(block, name)
            attn_name = name
            break

    if attn_module is None:
        print("[WARNING] Could not find attention module")
        return False

    print(f"[OK] Found attention module: block.{attn_name}")
    print(f"Attention type: {type(attn_module).__name__}")

    # Attention 모듈 상세
    print("\nAttention module attributes:")
    for attr in dir(attn_module):
        if not attr.startswith('_'):
            obj = getattr(attn_module, attr)
            if hasattr(obj, 'parameters'):
                # Linear layer 등
                if hasattr(obj, 'weight'):
                    shape = tuple(obj.weight.shape)
                    print(f"  [Linear] {attr}: {shape}")
                else:
                    print(f"  [Module] {attr}: {type(obj).__name__}")

    # Hook 가능 여부 확인
    print("\n" + "-" * 60)
    print("Hook Compatibility Check")
    print("-" * 60)

    has_to_q = hasattr(attn_module, 'to_q')
    has_to_k = hasattr(attn_module, 'to_k')
    has_to_v = hasattr(attn_module, 'to_v')
    has_num_heads = hasattr(attn_module, 'num_heads')

    print(f"  to_q: {'✓' if has_to_q else '✗'}")
    print(f"  to_k: {'✓' if has_to_k else '✗'}")
    print(f"  to_v: {'✓' if has_to_v else '✗'}")
    print(f"  num_heads: {'✓' if has_num_heads else '✗'}")

    if has_num_heads:
        print(f"    → num_heads = {attn_module.num_heads}")

    if has_to_q and has_to_k:
        print("\n[OK] Hook compatible! Can capture Q, K")

        # Q, K shape 확인
        q_weight = attn_module.to_q.weight
        k_weight = attn_module.to_k.weight
        print(f"  Q weight shape: {tuple(q_weight.shape)}")
        print(f"  K weight shape: {tuple(k_weight.shape)}")

        # head_dim 계산
        hidden_dim = q_weight.shape[0]
        num_heads = getattr(attn_module, 'num_heads', 32)
        head_dim = hidden_dim // num_heads
        print(f"  Estimated head_dim: {head_dim}")
    else:
        print("\n[WARNING] May need different hook approach")
        print("Check attention forward method implementation")

    # 요약
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
Model: Z-Image-Turbo
Architecture: S3-DiT
Blocks: {len(blocks)}
Attention module: block.{attn_name}
Hook compatible: {'Yes' if has_to_q and has_to_k else 'Needs investigation'}

Recommended hook layers: [10, 15, 20] (middle layers)

Next step: Run test_step2_attention_capture.py
""")

    return True


def main():
    success = explore_model_structure()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
