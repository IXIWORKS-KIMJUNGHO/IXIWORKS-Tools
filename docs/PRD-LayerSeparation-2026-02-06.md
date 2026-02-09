# PRD: StoryBoard 레이어 분리 이미지 생성

| 항목 | 내용 |
|------|------|
| **작성자** | IXIWORKS |
| **작성일** | 2026-02-06 |
| **버전** | 1.1 |
| **상태** | Draft |

---

## Changelog

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.1 | 2026-02-06 | Z-Image 모델 분석 결과 반영, KSamplerLayered 통합 노드 구조로 변경 |
| 1.0 | 2026-02-06 | 초안 작성 |

---

## Executive Summary

### 해결하려는 문제
현재 StoryBoard 워크플로우에서 생성된 이미지는 단일 레이어로 출력된다. 애니메틱스(카메라 이동, 패럴랙스 효과 등) 제작 시 피사체와 배경을 수동으로 분리해야 하는 비효율이 발생한다.

### 제안 솔루션
이미지 생성 과정에서 **Attention Map을 추출**하여 프롬프트 내 캐릭터/오브젝트별 마스크를 자동 생성하고, 이를 기반으로 레이어를 분리 출력한다.

### 기대 효과
- 애니메틱스 제작 시간 대폭 단축
- 스타일라이즈드 이미지(잉크 스케치, 수채화 등)에서도 정확한 분리
- StoryBoard JSON과 자연스러운 연동

---

## Background & Context

### 현재 상태
- StoryBoard JSON → 프롬프트 생성 → 이미지 생성 (단일 레이어)
- 애니메틱스 작업 시 Photoshop 등에서 수동 분리 필요
- 스타일라이즈드 이미지는 경계가 불명확해 분리 어려움

### 기존 솔루션 검토

| 솔루션 | 방식 | 한계 |
|--------|------|------|
| **Qwen-Image-Layered** | 생성 후 AI 분해 | 스타일라이즈드 이미지 분리 어려움 |
| **SAM 단독** | 세그멘테이션 | 캐릭터 식별 불가 (이름 모름) |
| **수동 분리** | Photoshop | 시간 소요, 반복 작업 |

### 선택한 접근법: Attention Map 기반
- 생성 과정에서 프롬프트 토큰별 attention weight 추출
- 캐릭터 이름 토큰 → 해당 캐릭터 위치 히트맵
- 스타일과 무관하게 **생성 의도 기반** 분리 가능

---

## Goals & Objectives

### Primary Goals
1. StoryBoard 워크플로우에서 **자동 레이어 분리** 지원
2. **PNG + JSON** 및 **PSD** 형식 출력
3. 스타일라이즈드 이미지에서도 **정확한 분리**
4. 기존 워크플로우 **최소 변경**으로 통합

### Success Metrics
| 지표 | 목표 |
|------|------|
| 분리 정확도 | 90% 이상 (수동 보정 최소화) |
| 추가 처리 시간 | 이미지당 +5초 이내 |
| 지원 스타일 | 포토리얼, 잉크 스케치, 수채화, 라인아트 |

---

## User Stories

### US-01: 애니메틱스 제작자
> "스토리보드 이미지를 생성하면 캐릭터와 배경이 자동으로 분리되어, After Effects에서 바로 패럴랙스 작업을 할 수 있다."

### US-02: 스토리보드 아티스트
> "잉크 스케치 스타일로 생성해도 캐릭터별로 레이어가 분리되어, 개별 수정이 가능하다."

### US-03: 워크플로우 설계자
> "기존 StoryBoard JSON에 레이어 정보를 추가하면, 자동으로 해당 구조로 분리된다."

---

## Technical Architecture

### 대상 모델: Z-Image-Turbo

| 항목 | 값 |
|------|-----|
| **아키텍처** | S3-DiT (Scalable Single-Stream DiT) |
| **레이어 수** | 30개 Transformer 블록 |
| **은닉 차원** | 3,840 |
| **Attention 헤드** | 32개 |
| **파라미터** | 6.15B |
| **Attention Backend** | SDPA (기본), Flash Attention 2/3 (옵션) |
| **텍스트 인코더** | Qwen3 4B |

### 토큰 처리 방식

```
[텍스트 토큰] + [시각 의미 토큰] + [이미지 VAE 토큰]
                    ↓
            시퀀스 레벨에서 concat
                    ↓
        30개 Transformer 블록 통과
        (각 블록에서 Self-Attention)
                    ↓
    "매 블록에서 깊은 크로스 모달 상호작용"
```

### Attention 구현 분석 (DiffSynth-Studio 기반)

**핵심 클래스:**
- `ZImageDiT` - 메인 트랜스포머
- `ZImageTransformerBlock` - 각 레이어 블록
- `Attention` - Self-Attention 구현

**Attention Forward 흐름:**
```python
def forward(self, hidden_states, freqs_cis, attention_mask):
    # 1. Q, K, V 계산 ← Hook 가능!
    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)

    # 2. RMSNorm (QK-Norm)
    query = self.norm_q(query)
    key = self.norm_k(key)

    # 3. RoPE (3D Unified RoPE) 적용
    # ...

    # 4. SDPA 호출 ← weights 직접 반환 안 함!
    hidden_states = attention_forward(query, key, value, ...)
```

### SDPA 제한 및 해결책

**문제**: PyTorch `scaled_dot_product_attention`은 attention weights를 반환하지 않음

**해결책**: Q, K를 Hook으로 캡처하여 직접 계산

```python
# Attention weights 직접 계산
attention_weights = torch.softmax(
    query @ key.transpose(-2, -1) / math.sqrt(head_dim),
    dim=-1
)
# 텍스트 토큰 → 이미지 토큰 영역만 추출
text_to_image_weights = attention_weights[text_indices, image_indices]
```

---

## Node Architecture (Updated)

### 기존 계획의 문제점
- Model Wrapper 여러 개 (ProgressTracker, AttentionExtractor) → Hook 충돌 가능
- 워크플로우 복잡

### 새로운 구조: KSamplerLayered 통합 노드

**단일 노드에서 모든 처리:**

```
┌─────────────────────────────────────────────────────────────────┐
│  KSamplerLayered                                                │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│  - model, positive, negative, latent                            │
│  - seed, steps, cfg, sampler_name, scheduler                    │
│  - extract_tokens: "Mina,Robot" (분리 대상 캐릭터)              │
│  - redis_url, job_id (progress 추적용, 선택)                    │
│  - attention_layers: "10,15,20" (추출할 레이어, 선택)           │
├─────────────────────────────────────────────────────────────────┤
│  Outputs:                                                       │
│  - latent (기존과 동일)                                         │
│  - attention_maps (토큰별 히트맵 딕셔너리)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 업데이트된 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│  StoryBoard JSON (캐릭터 정보 포함)                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Build Prompt + Extract Token Names                             │
│  예: "Mina is walking, Robot stands nearby"                     │
│  추출: extract_tokens = "Mina,Robot"                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  KSamplerLayered (통합 노드)                                    │
│  - Sampling 수행                                                │
│  - 내부에서 Attention 레이어에 Hook 등록                         │
│  - Q, K 캡처 → weights 계산 → 토큰별 히트맵 생성                │
│  - Progress 추적 (선택)                                         │
└─────────────────────────────────────────────────────────────────┘
          ↓                              ↓
      latent                      attention_maps
          ↓                              ↓
┌──────────────────┐     ┌────────────────────────────────────────┐
│  VAE Decode      │     │  AttentionToMask                       │
│                  │     │  - 히트맵 → threshold → 이진 마스크     │
│                  │     │  - (선택) SAM 정밀화                    │
└──────────────────┘     └────────────────────────────────────────┘
          ↓                              ↓
       IMAGE                          MASKS
          └──────────────┬───────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  LayerSeparator                                                 │
│  - 마스크 기반 레이어 분리                                       │
│  - ControlNet Depth로 레이어 순서 결정                           │
│  - (선택) 배경 Inpainting                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LayerExporter                                                  │
│  - PNG 여러 장 + composition.json                                │
│  - PSD (레이어 포함)                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Functional Requirements

### Must-Have (P0)

| ID | 요구사항 | 설명 |
|----|----------|------|
| F-01 | KSamplerLayered 노드 | Sampling + Attention 추출 통합 노드 |
| F-02 | Q, K Hook | ZImageDiT Attention 레이어에서 Q, K 캡처 |
| F-03 | Attention weights 계산 | Q @ K^T / sqrt(d) → softmax |
| F-04 | 토큰-히트맵 변환 | 텍스트→이미지 attention을 이미지 해상도로 reshape |
| F-05 | AttentionToMask 노드 | 히트맵을 이진 마스크로 변환 |
| F-06 | LayerSeparator 노드 | 마스크 기반 레이어 분리 |
| F-07 | PNG + JSON 출력 | 레이어별 투명 PNG + composition.json |

### Should-Have (P1)

| ID | 요구사항 | 설명 |
|----|----------|------|
| F-08 | SAM 마스크 정밀화 | Attention 히트맵을 SAM으로 정밀하게 다듬기 |
| F-09 | PSD 출력 | Adobe 호환 PSD 파일 출력 |
| F-10 | 뎁스 기반 레이어 순서 | ControlNet Depth로 앞/뒤 순서 자동 결정 |
| F-11 | JSON 뎁스 오버라이드 | StoryBoard JSON에서 레이어 순서 수동 지정 |
| F-12 | Progress 통합 | Redis 기반 progress 추적 (기존 기능 통합) |

### Could-Have (P2)

| ID | 요구사항 | 설명 |
|----|----------|------|
| F-13 | 배경 Inpainting | 피사체 제거 후 배경 자동 채우기 |
| F-14 | 다중 피사체 분리 | 3개 이상 피사체 개별 분리 |
| F-15 | 레이어 미리보기 | UI에서 분리 결과 미리보기 |
| F-16 | 레이어별 attention 선택 | 특정 레이어만 attention 추출 |

---

## Non-Functional Requirements

### Performance
- 추가 처리 시간: 1024x1024 이미지 기준 **5초 이내**
- 메모리 오버헤드: **2GB 이내** 추가
- Attention 저장: 선택적 레이어만 (메모리 최적화)

### Compatibility
- ComfyUI 최신 버전 지원
- Z-Image-Turbo 모델 지원
- DiffSynth-Studio 기반 구현 (ComfyUI_DSS_Wrapper 호환)
- 기존 StoryBoard 워크플로우와 호환

### Output Format

**composition.json 스키마:**
```json
{
  "version": "1.0",
  "source_image": "scene_01.png",
  "canvas": {
    "width": 1024,
    "height": 1024
  },
  "layers": [
    {
      "name": "Mina",
      "file": "Mina.png",
      "depth": 0.3,
      "bounds": { "x": 100, "y": 200, "width": 300, "height": 500 },
      "token_indices": [3, 4],
      "attention_layers_used": [10, 15, 20]
    },
    {
      "name": "Robot",
      "file": "Robot.png",
      "depth": 0.6,
      "bounds": { "x": 500, "y": 180, "width": 280, "height": 520 },
      "token_indices": [8],
      "attention_layers_used": [10, 15, 20]
    },
    {
      "name": "background",
      "file": "background.png",
      "depth": 1.0,
      "bounds": { "x": 0, "y": 0, "width": 1024, "height": 1024 },
      "inpainted": true
    }
  ]
}
```

---

## Implementation Plan

### Phase 1: Attention 추출 PoC
- [ ] ZImageDiT 구조 분석 (DiffSynth-Studio 소스코드)
- [ ] Attention 레이어 Hook 테스트
- [ ] Q, K 캡처 → weights 계산 검증
- [ ] 특정 토큰 히트맵 시각화

### Phase 2: KSamplerLayered 노드
- [ ] 기존 KSampler 로직 통합
- [ ] Attention Hook 등록/해제 로직
- [ ] 토큰 인덱스 매핑 (토크나이저 분석)
- [ ] attention_maps 출력 구현

### Phase 3: 마스크 & 분리
- [ ] AttentionToMask 노드 개발
- [ ] Threshold 기반 마스크 생성
- [ ] LayerSeparator 노드 개발
- [ ] LayerExporter 노드 개발

### Phase 4: 정밀화 & 확장
- [ ] SAM 연동
- [ ] PSD 출력 (psd-tools)
- [ ] ControlNet Depth 연동
- [ ] 배경 Inpainting

### Phase 5: StoryBoard 통합
- [ ] JSON 스키마 확장 (layers 필드)
- [ ] 기존 워크플로우 업데이트
- [ ] 문서화

---

## Risks & Mitigation

| 리스크 | 영향 | 확률 | 완화 방안 |
|--------|------|------|-----------|
| SDPA가 weights 미반환 | 직접 계산 필요 | **해결됨** | Q @ K^T로 직접 계산 |
| Attention 해상도 낮음 | 마스크 품질 저하 | 중 | 여러 레이어 평균, SAM 정밀화 |
| 토큰 위치 파악 어려움 | 잘못된 마스크 | 낮 | 토크나이저 분석, 디버그 시각화 |
| 스타일에 따른 편차 | 일관성 저하 | 중 | 스타일별 threshold 튜닝 |
| VRAM 부족 | 실행 불가 | 낮 | 선택적 레이어만 저장, 메모리 최적화 |
| Hook 충돌 (Progress 등) | 오작동 | **해결됨** | KSamplerLayered로 통합 |

---

## Open Questions

### 해결됨
- ~~SDPA에서 attention weights 접근 가능한가?~~ → Q, K 캡처 후 직접 계산
- ~~여러 Hook 충돌 문제~~ → 통합 노드로 해결

### 남은 질문
1. **최적 레이어 선택**: 30개 중 어느 레이어가 가장 의미있는 마스크를 주는가?
   - 예상: 중간 레이어 (10-20) 유력
   - 방법: 실험으로 확인

2. **스텝별 가중치**: 초기 스텝 vs 후기 스텝 attention 중 어느 것이 더 정확한가?

3. **복합 토큰 처리**: "Mina"가 ["Mi", "na"]로 분리될 경우 어떻게 합칠 것인가?
   - 방법: 토큰 범위의 attention 평균

4. **겹치는 피사체**: 두 캐릭터가 겹칠 때 마스크 우선순위는?
   - 방법: depth 값 기반 또는 attention 강도 기반

---

## Dependencies

- **Z-Image-Turbo**: HuggingFace `Tongyi-MAI/Z-Image-Turbo`
- **DiffSynth-Studio**: Attention 구현 참고 (`diffsynth.models.z_image_dit`)
- **ComfyUI_DSS_Wrapper**: DiffSynth-Studio ComfyUI 래퍼 (선택)
- **SAM (Segment Anything)**: 마스크 정밀화용
- **psd-tools**: PSD 출력용 Python 라이브러리
- **ControlNet Depth**: 뎁스 기반 레이어 순서용

---

## Appendix

### 참고 자료
- [Z-Image-Turbo (HuggingFace)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image 논문 (arXiv)](https://arxiv.org/abs/2511.22699)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [ComfyUI_DSS_Wrapper](https://github.com/quinteroac/ComfyUI_DSS_Wrapper)
- [Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered) - 대안 접근법 참고

### Z-Image 모델 구조 요약

| 컴포넌트 | 설명 |
|----------|------|
| **S3-DiT** | Scalable Single-Stream DiT |
| **토큰 처리** | 텍스트 + 시각 의미 + 이미지 VAE → 단일 시퀀스 |
| **Attention** | QK-Norm + Sandwich-Norm + 3D Unified RoPE |
| **레이어** | 30개 Transformer 블록 |
| **Attention Backend** | SDPA (기본), Flash Attention (옵션) |

### 용어 정의
| 용어 | 설명 |
|------|------|
| Attention Map | 모델이 특정 토큰에 집중하는 이미지 영역을 나타내는 히트맵 |
| S3-DiT | Scalable Single-Stream DiT, 텍스트와 이미지 토큰을 하나의 시퀀스로 처리 |
| SDPA | Scaled Dot Product Attention, PyTorch 기본 attention 구현 |
| SAM | Segment Anything Model, Meta의 범용 세그멘테이션 모델 |
| 애니메틱스 | 스토리보드를 기반으로 카메라 움직임 등을 추가한 간이 애니메이션 |
| Hook | 모델 실행 중 특정 지점에서 데이터를 캡처하거나 수정하는 콜백 |
