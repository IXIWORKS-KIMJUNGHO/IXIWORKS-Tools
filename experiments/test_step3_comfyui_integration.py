"""
Step 3: ComfyUI 통합 테스트
===========================

목적: ComfyUI에서 layer_nodes.py 노드들이 정상 작동하는지 확인

이 스크립트는 직접 실행하는 것이 아니라,
ComfyUI 환경에서 테스트하기 위한 가이드입니다.

테스트 방법:
1. IXIWORKS-Tools를 ComfyUI/custom_nodes/에 링크 또는 복사
2. ComfyUI 재시작
3. 아래 워크플로우 JSON을 ComfyUI에서 로드
4. 결과 확인
"""

# =============================================================================
# ComfyUI 테스트 워크플로우 (JSON)
# =============================================================================

COMFYUI_TEST_WORKFLOW = """
{
  "last_node_id": 10,
  "last_link_id": 10,
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [300, 100],
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "MODEL", "type": "MODEL", "links": [1]},
        {"name": "CLIP", "type": "CLIP", "links": [2]},
        {"name": "VAE", "type": "VAE", "links": [3]}
      ],
      "properties": {},
      "widgets_values": ["z-image-turbo.safetensors"]
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "pos": [350, 0],
      "size": [300, 100],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {"name": "clip", "type": "CLIP", "link": 2}
      ],
      "outputs": [
        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [4]}
      ],
      "properties": {},
      "widgets_values": ["A warrior named Mina and a robot named Bot standing in a forest, ink sketch style"]
    },
    {
      "id": 3,
      "type": "CLIPTextEncode",
      "pos": [350, 150],
      "size": [300, 100],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {"name": "clip", "type": "CLIP", "link": null}
      ],
      "outputs": [
        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [5]}
      ],
      "properties": {},
      "widgets_values": ["blurry, low quality"]
    },
    {
      "id": 4,
      "type": "EmptyLatentImage",
      "pos": [350, 300],
      "size": [300, 100],
      "flags": {},
      "order": 3,
      "mode": 0,
      "outputs": [
        {"name": "LATENT", "type": "LATENT", "links": [6]}
      ],
      "properties": {},
      "widgets_values": [512, 512, 1]
    },
    {
      "id": 5,
      "type": "KSamplerLayered",
      "pos": [700, 100],
      "size": [350, 300],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [
        {"name": "model", "type": "MODEL", "link": 1},
        {"name": "positive", "type": "CONDITIONING", "link": 4},
        {"name": "negative", "type": "CONDITIONING", "link": 5},
        {"name": "latent_image", "type": "LATENT", "link": 6}
      ],
      "outputs": [
        {"name": "latent", "type": "LATENT", "links": [7]},
        {"name": "attention_maps", "type": "ATTENTION_MAPS", "links": [8]}
      ],
      "properties": {},
      "widgets_values": [
        12345,
        20,
        7.0,
        "euler",
        "normal",
        1.0,
        "Mina,Bot",
        "10,15,20",
        "",
        ""
      ]
    },
    {
      "id": 6,
      "type": "VAEDecode",
      "pos": [1100, 0],
      "size": [200, 100],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {"name": "samples", "type": "LATENT", "link": 7},
        {"name": "vae", "type": "VAE", "link": 3}
      ],
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [9]}
      ],
      "properties": {}
    },
    {
      "id": 7,
      "type": "AttentionToMask",
      "pos": [1100, 150],
      "size": [250, 150],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [
        {"name": "attention_maps", "type": "ATTENTION_MAPS", "link": 8}
      ],
      "outputs": [
        {"name": "combined_mask", "type": "MASK", "links": []},
        {"name": "individual_masks", "type": "MASK_LIST", "links": [10]}
      ],
      "properties": {},
      "widgets_values": [0.5, 5, 0, true]
    },
    {
      "id": 8,
      "type": "LayerSeparator",
      "pos": [1400, 100],
      "size": [250, 150],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [
        {"name": "image", "type": "IMAGE", "link": 9},
        {"name": "mask_list", "type": "MASK_LIST", "link": 10}
      ],
      "outputs": [
        {"name": "layers", "type": "LAYER_LIST", "links": [11]}
      ],
      "properties": {},
      "widgets_values": [true, 3]
    },
    {
      "id": 9,
      "type": "LayerExporter",
      "pos": [1700, 100],
      "size": [250, 150],
      "flags": {},
      "order": 8,
      "mode": 0,
      "inputs": [
        {"name": "layers", "type": "LAYER_LIST", "link": 11}
      ],
      "outputs": [
        {"name": "output_path", "type": "STRING", "links": []}
      ],
      "properties": {},
      "widgets_values": ["output/layers", "test_scene", false, true]
    },
    {
      "id": 10,
      "type": "PreviewImage",
      "pos": [1100, 350],
      "size": [200, 200],
      "flags": {},
      "order": 9,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 9}
      ],
      "properties": {}
    }
  ],
  "links": [
    [1, 1, 0, 5, 0, "MODEL"],
    [2, 1, 1, 2, 0, "CLIP"],
    [3, 1, 2, 6, 1, "VAE"],
    [4, 2, 0, 5, 1, "CONDITIONING"],
    [5, 3, 0, 5, 2, "CONDITIONING"],
    [6, 4, 0, 5, 3, "LATENT"],
    [7, 5, 0, 6, 0, "LATENT"],
    [8, 5, 1, 7, 0, "ATTENTION_MAPS"],
    [9, 6, 0, 8, 0, "IMAGE"],
    [10, 7, 1, 8, 1, "MASK_LIST"],
    [11, 8, 0, 9, 0, "LAYER_LIST"]
  ],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}
"""

# =============================================================================
# 테스트 체크리스트
# =============================================================================

TEST_CHECKLIST = """
## ComfyUI 통합 테스트 체크리스트

### 1. 설치 확인
- [ ] IXIWORKS-Tools가 custom_nodes에 있는지 확인
- [ ] ComfyUI 시작 시 에러 없이 로드되는지 확인
- [ ] IXIWORKS/Layer 카테고리에 노드들이 보이는지 확인:
  - [ ] KSampler Layered (Attention Extract)
  - [ ] Attention To Mask
  - [ ] Layer Separator
  - [ ] Layer Exporter

### 2. 기본 연결 테스트
- [ ] KSamplerLayered 노드가 MODEL, CONDITIONING, LATENT 입력을 받는지
- [ ] latent, attention_maps 출력이 나오는지
- [ ] AttentionToMask가 attention_maps를 받아 마스크 출력하는지
- [ ] LayerSeparator가 IMAGE + MASK_LIST로 레이어 분리하는지
- [ ] LayerExporter가 파일 저장하는지

### 3. Attention 캡처 테스트
- [ ] extract_tokens에 "Mina,Bot" 입력
- [ ] 생성 완료 후 attention_maps가 비어있지 않은지
- [ ] 콘솔 로그에 "[AttentionHook] Captured layer X" 메시지 확인

### 4. 마스크 품질 테스트
- [ ] 생성된 마스크가 캐릭터 위치와 대략 일치하는지
- [ ] threshold 조절 시 마스크 크기가 변하는지
- [ ] blur_radius 조절 시 마스크 경계가 부드러워지는지

### 5. 출력 테스트
- [ ] output/layers/ 폴더에 파일 생성되는지
- [ ] PNG 파일들이 투명 배경으로 저장되는지
- [ ] composition.json 파일이 올바른 구조인지:
  ```json
  {
    "version": "1.0",
    "layers": [
      {"name": "Mina", "file": "...", "bounds": {...}},
      {"name": "Bot", "file": "...", "bounds": {...}},
      {"name": "background", "file": "...", "bounds": {...}}
    ]
  }
  ```

### 6. 에러 처리 테스트
- [ ] extract_tokens가 비어있을 때 기본 동작 (attention_maps = {})
- [ ] 잘못된 토큰 이름일 때 경고 로그
- [ ] 모델 구조가 다를 때 graceful fallback

### 7. 성능 테스트
- [ ] 기존 KSampler 대비 추가 시간 측정
- [ ] VRAM 사용량 확인
- [ ] attention_layers 수에 따른 메모리 변화

## 예상 이슈 및 해결방법

### Issue 1: Hook이 등록되지 않음
원인: 모델 구조가 예상과 다름
해결: test_step1_model_structure.py로 실제 구조 확인 후 layer_nodes.py 수정

### Issue 2: Attention shape 불일치
원인: 토큰 수 추정이 틀림
해결: 실제 seq_len 확인 후 text_token_count, visual_token_count 조정

### Issue 3: 히트맵이 uniform함
원인: 잘못된 토큰 인덱스
해결: 토크나이저 출력 확인하여 정확한 토큰 위치 파악

### Issue 4: 마스크가 너무 작거나 큼
원인: threshold가 데이터에 맞지 않음
해결: Otsu threshold 사용 또는 수동 조절
"""

def save_workflow():
    """테스트 워크플로우 JSON 저장"""
    import os
    import json

    output_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_path = os.path.join(output_dir, "test_workflow_layer_separation.json")

    workflow = json.loads(COMFYUI_TEST_WORKFLOW)
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)

    print(f"[OK] Saved test workflow to: {workflow_path}")

    checklist_path = os.path.join(output_dir, "test_checklist.md")
    with open(checklist_path, 'w') as f:
        f.write(TEST_CHECKLIST)

    print(f"[OK] Saved test checklist to: {checklist_path}")


if __name__ == "__main__":
    save_workflow()
    print("\n" + "=" * 60)
    print("ComfyUI Integration Test Guide")
    print("=" * 60)
    print("""
1. Copy IXIWORKS-Tools to ComfyUI/custom_nodes/

2. Restart ComfyUI

3. Load the test workflow:
   experiments/test_workflow_layer_separation.json

4. Follow the test checklist:
   experiments/test_checklist.md

5. Check console logs for attention capture messages

6. Verify output files in output/layers/
""")
