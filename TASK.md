# ControlNet Preprocessor 통합 노드 개발

## 개요

하나의 드롭다운 노드로 Canny, Depth, Lineart, Pose 등 17종 ControlNet 전처리기를 선택/실행하는 통합 노드.

---

## 지원 전처리기

| # | ID | 설명 | 모델 필요 |
|---|---|---|---|
| 1 | canny | 엣지 검출 (OpenCV) | X |
| 2 | depth_anything_v2 | 깊이맵 (Depth Anything V2) | O |
| 3 | lineart_realistic | 라인아트 (사실적) | O |
| 4 | lineart_coarse | 라인아트 (거친) | O |
| 5 | lineart_anime | 라인아트 (애니메) | O |
| 6 | softedge_hed | 소프트엣지 (HED) | O |
| 7 | softedge_pidinet | 소프트엣지 (PiDiNet) | O |
| 8 | softedge_teed | 소프트엣지 (TEED) | O |
| 9 | scribble_hed | 스크리블 (HED) | O |
| 10 | scribble_pidinet | 스크리블 (PiDiNet) | O |
| 11 | openpose | 포즈 (body) | O |
| 12 | openpose_full | 포즈 (body+hand+face) | O |
| 13 | dwpose | 포즈 (DWPose) | O |
| 14 | mlsd | 직선 검출 | O |
| 15 | normal_bae | 노멀맵 | O |
| 16 | shuffle | 콘텐츠 셔플 | X |

---

## 노드 입출력

### 입력
| 파라미터 | 타입 | 기본값 | 필수 | 비고 |
|----------|------|--------|------|------|
| image | IMAGE | - | O | BHWC float32 [0,1] |
| preprocessor | 드롭다운 | canny | O | 17종 선택 |
| resolution | INT | 512 | O | 256~2048, step 64 |
| low_threshold | INT | 100 | X | Canny 전용, 0~255 |
| high_threshold | INT | 200 | X | Canny 전용, 0~255 |

### 출력
| 파라미터 | 타입 | 비고 |
|----------|------|------|
| image | IMAGE | BHWC float32 [0,1], 항상 3채널 RGB |

---

## 작업 항목

### Step 1. controlnet_nodes.py 생성
- [ ] PREPROCESSOR_REGISTRY 정의 (17종 전처리기 매핑)
- [ ] ControlNetPreprocessorNode 클래스 작성
  - [ ] INPUT_TYPES (image, preprocessor, resolution, threshold)
  - [ ] _get_detector() 캐싱 메서드
  - [ ] preprocess() 메인 함수 (텐서 변환 + 배치 처리)
- [ ] NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS

### Step 2. js/controlnet.js 생성
- [ ] preprocessor 드롭다운 변경 시 Canny 위젯 동적 표시/숨김
- [ ] 그래프 로드 시 위젯 상태 복원 (onConfigure)

### Step 3. __init__.py 수정
- [ ] controlnet_nodes import 추가
- [ ] NODE_CLASS_MAPPINGS 병합
- [ ] NODE_DISPLAY_NAME_MAPPINGS 병합

### Step 4. 의존성 추가
- [ ] requirements.txt에 controlnet-aux 추가
- [ ] pyproject.toml dependencies에 controlnet-aux 추가

### Step 5. 검증
- [ ] ComfyUI 시작 시 import 에러 없는지 확인
- [ ] 드롭다운 17종 전처리기 표시 확인
- [ ] Canny 위젯 동적 표시/숨김 확인
- [ ] Canny 이미지 처리 → 3채널 RGB 출력
- [ ] depth_midas 처리 → 모델 자동 다운로드 + depth map 출력
- [ ] 배치 이미지(B>1) 처리 확인
- [ ] 워크플로우 저장/로드 시 설정 유지

---

## 기술 사항

### 라이브러리
- `controlnet_aux` (pip install controlnet-aux)
- 모델: HuggingFace `lllyasviel/Annotators`에서 자동 다운로드

### 캐싱 전략
- 클래스 변수 `_detector_cache` dict에 detector 인스턴스 저장
- 키: detector 클래스명 (같은 클래스 공유하는 전처리기끼리 캐시 공유)
- ComfyUI 프로세스 종료까지 유지

### 텐서 변환 (image_nodes.py 패턴)
```
torch BHWC [0,1] → numpy*255 uint8 → PIL → detector → PIL.convert("RGB") → numpy/255 → torch [0,1]
```

### 특이 사항
- TEED: `image_resolution` 파라미터 미지원 → kwargs에서 제거
- OpenPose: body만 vs full(body+hand+face) 분리
- DWPose: onnxruntime 추가 필요 가능

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-02 | 태스크 문서 작성 |
