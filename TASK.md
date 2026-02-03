# LoRA Loader Advanced 노드 개발

## 개요

ComfyUI Hook Keyframe 네이티브 기능을 활용하여, step별 LoRA 강도를 점진적으로 조절하는 올인원 LoRA 로더 노드.
DiffSynth ControlNet Advanced와 동일한 인터랙티브 그래프 위젯 포함.

---

## 아키텍처

### Hook Keyframe 방식 (ComfyUI 네이티브)

LoRA는 가중치를 직접 수정하므로 ControlNet처럼 per-step 패치 래핑이 불가.
ComfyUI의 Hook 시스템을 활용하여 step별 강도 스케줄링 구현.

**내부 동작:**
1. LoRA 파일 로드
2. CLIP: 전통적 `add_patches`로 고정 strength 적용
3. MODEL: `comfy.hooks.create_hook_lora()`로 Hook 생성 → Keyframe 스케줄링
4. HOOKS 출력 → 사용자가 Conditioning 노드에 연결

**워크플로우:**
```
[LoRA Loader Advanced] ─ MODEL ─→ [KSampler]
                       ─ CLIP  ─→ [CLIPTextEncode] ─→ [Cond Set Props] ─→ [KSampler]
                       ─ HOOKS ─────────────────────→ [Cond Set Props]
```

### 핵심 API

| ComfyUI API | 용도 |
|---|---|
| `comfy.hooks.create_hook_lora(lora, strength_model, 0)` | MODEL용 Hook 생성 |
| `comfy.hooks.HookKeyframeGroup` | Keyframe 그룹 생성 |
| `comfy.hooks.HookKeyframe(strength, start_percent)` | 개별 Keyframe |
| `clip.clone() + clip.add_patches(loaded, strength_clip)` | CLIP 고정 strength 적용 |

---

## 노드 입출력

### 입력

| 파라미터 | 타입 | 기본값 | 범위 | 비고 |
|----------|------|--------|------|------|
| model | MODEL | - | - | 필수 |
| clip | CLIP | - | - | 필수 |
| lora_name | 드롭다운 | - | models/loras/ | 필수, folder_paths 사용 |
| strength_model | FLOAT | 1.0 | 0.0~2.0, step 0.01 | fade 시 max strength |
| strength_clip | FLOAT | 1.0 | 0.0~2.0, step 0.01 | 고정 (fade 미적용) |
| start | FLOAT | 0.0 | 0.0~1.0, step 0.01 | 시작 지점 (%) |
| end | FLOAT | 1.0 | 0.0~1.0, step 0.01 | 종료 지점 (%) |
| fade | 드롭다운 | none | none/fade out/fade in | fade 모드 |
| low | FLOAT | 0.0 | 0.0~2.0, step 0.01 | fade 시 min strength |

### 출력

| 파라미터 | 타입 | 비고 |
|----------|------|------|
| model | MODEL | 패스스루 (Hook이 sampling 시 동적 적용) |
| clip | CLIP | LoRA 적용됨 (strength_clip 고정) |
| hooks | HOOKS | Conditioning 노드에 연결 필요 |

---

## Keyframe 생성 로직

`strength_mult`는 base `strength_model`에 곱해지는 배율.

### fade = "none" (범위 내 일정 강도)
```
percent: 0.0     start     end      1.0
mult:    0.0 ─── 1.0 ───── 1.0 ─── 0.0
```

### fade = "fade out" (강 → 약)
```
percent: 0.0     start ─────── end      1.0
mult:    0.0 ─── 1.0 ───→ low/str ─── 0.0
         (10개 보간 keyframe)
```

### fade = "fade in" (약 → 강)
```
percent: 0.0     start ─────── end      1.0
mult:    0.0 ─── low/str ──→ 1.0 ─── 0.0
         (10개 보간 keyframe)
```

---

## JS 프론트엔드 (위젯 동작)

### fade = "none" 상태
| 위젯 | 표시 |
|------|------|
| lora_name | ✅ 항상 표시 |
| strength_model | ✅ 표시 |
| strength_clip | ✅ 항상 표시 |
| start / end | ✅ 항상 표시 |
| fade | ✅ 항상 표시 |
| low | ❌ 숨김 |
| strength_graph | ❌ 숨김 |

### fade = "fade out" / "fade in" 상태
| 위젯 | 표시 |
|------|------|
| lora_name | ✅ 항상 표시 |
| strength_model | ❌ 숨김 (그래프가 제어) |
| strength_clip | ✅ 항상 표시 |
| start / end | ✅ 항상 표시 |
| fade | ✅ 항상 표시 |
| low | ✅ 표시 |
| strength_graph | ✅ 표시 (인터랙티브) |

### 그래프 위젯
- `js/controlnet.js`의 DiffSynthControlnetAdvanced 그래프 코드 재사용
- 높이: 120px, 드래그 정밀도: 0.1
- 위젯 참조명: `strength_model` (기존 `strength` 대신)
- 색상: `#B39DDB` (보라색 계열, ControlNet과 동일)

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `lora_nodes.py` | LoraLoaderAdvanced 노드 클래스 (신규) |
| `js/lora.js` | 그래프 위젯 + fade 위젯 표시/숨김 (신규) |
| `__init__.py` | lora_nodes import 추가 (수정) |

---

## 작업 항목

### Step 1. `lora_nodes.py` 생성
- [ ] LoraLoaderAdvanced 클래스 작성
  - [ ] INPUT_TYPES 정의 (model, clip, lora_name, strength_model, strength_clip, start, end, fade, low)
  - [ ] RETURN_TYPES = ("MODEL", "CLIP", "HOOKS")
  - [ ] LoRA 파일 로드 + 캐싱 (LoraLoader 패턴)
  - [ ] CLIP에 LoRA 적용 (고정 strength_clip)
  - [ ] Hook 생성: `comfy.hooks.create_hook_lora(lora, strength_model, 0)`
  - [ ] Keyframe 생성: fade 모드별 보간 로직
  - [ ] HookKeyframeGroup에 Keyframe 추가 + Hook에 설정
- [ ] NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS

### Step 2. `js/lora.js` 생성
- [ ] `IXIWORKS.LoraLoaderAdvanced` 확장 등록
- [ ] fade 드롭다운 콜백 → `_updateFadeWidgets` 호출
- [ ] `_updateFadeWidgets`: strength_model/low/graph 표시/숨김 제어
- [ ] 인터랙티브 그래프 위젯 (controlnet.js에서 복사 + 수정)
  - [ ] `_getGraphParams`: `strength` → `strength_model`로 위젯명 변경
  - [ ] `draw`: 동일한 보라색 그래프 렌더링
  - [ ] `mouse`: 동일한 드래그 인터랙션
  - [ ] `computeSize`: 120px 높이
- [ ] `onConfigure`: 그래프 로드 시 위젯 상태 복원

### Step 3. `__init__.py` 수정
- [ ] `from .lora_nodes import ...` 추가
- [ ] NODE_CLASS_MAPPINGS 병합
- [ ] NODE_DISPLAY_NAME_MAPPINGS 병합

### Step 4. 검증
- [ ] ComfyUI 시작 시 import 에러 없는지 확인
- [ ] 노드 목록에 "LoRA Loader Advanced (IXIWORKS)" 표시
- [ ] lora_name 드롭다운에 models/loras/ 파일 목록 표시
- [ ] fade=none: strength_model 위젯만 표시, 그래프 숨김
- [ ] fade=fade out/in: 그래프 표시, strength_model 숨김
- [ ] 그래프 드래그 → strength_model/low/start/end 값 변경
- [ ] HOOKS 출력 → PairConditioningSetProperties 연결 가능
- [ ] 워크플로우 저장/로드 시 설정 유지

### Step 5. 버전업 + 배포
- [ ] pyproject.toml 버전 bump
- [ ] 커밋 및 푸쉬
- [ ] `comfy node publish`

---

## 참고: 기존 코드 패턴

### LoRA 로드 (ComfyUI LoraLoader)
```python
lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
```

### Hook 생성 (ComfyUI nodes_hooks.py)
```python
hooks = comfy.hooks.create_hook_lora(lora=lora, strength_model=strength_model, strength_clip=0)

hook_kf = comfy.hooks.HookKeyframeGroup()
hook_kf.add(comfy.hooks.HookKeyframe(strength=mult, start_percent=pct))

for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight):
    hook.hook_keyframe = hook_kf
```

### 위젯 숨김/표시 (JS)
```javascript
// 숨김
w.type = "hidden";
w.computeSize = () => [0, -4];
// 표시
w.type = "number";
w.computeSize = undefined;
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-03 | 태스크 문서 작성 |
