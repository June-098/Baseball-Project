# 05. 비전 극대화 + 2026-07 로컬 멀티모달 LLM 로드맵

기준 컷오프: 2026-07-31  
대상 하드웨어: Apple M5 Max, GPU 40코어, 통합 메모리 128GB

## 결론부터

이 제품은 **LLM이 영상을 보고 숫자를 만들어내는 앱**이 아니라, **결정론적 CV/기하 파이프라인이 측정값과 불확실성을 만들고 로컬 VLM이 근거를 검토·설명하는 앱**으로 설계해야 한다. 이 경계를 지켜야 야구 지표가 재현되고, 모델 교체 후에도 같은 입력을 비교할 수 있다.

권장 2-lane 구성은 다음과 같다.

| lane | 모델/형식 | 공식 사양 | 이 프로젝트의 역할 |
|---|---|---|---|
| fast multimodal | `google/gemma-4-12B-it-qat-q4_0-gguf` | Gemma 4 12B Unified, 11.95B dense, 256K context, text/image/audio 및 모델 카드상 video 입력, Apache-2.0, 공식 Q4_0 약 6.98GB | 촬영 품질 판정, 키프레임/짧은 클립 1차 설명, 리포트 초안, 로컬 UI 응답 |
| strong multimodal | `mlx-community/Qwen3.6-35B-A3B-4bit` (원본 `Qwen/Qwen3.6-35B-A3B`) | 총 35B/활성 3B MoE, native 262,144 context, vision encoder와 long-video 가이드, Apache-2.0, MLX 4-bit 약 20.4GB | 경계 사례 재검토, 여러 스윙 비교, 근거 종합, 최종 개인화 코칭 |

두 모델 모두 메모리에 들어가지만 처음부터 동시 상주는 권하지 않는다. fast lane 상주 + strong lane 요청 시 로드, 또는 둘을 직렬 실행해 메모리 대역폭 경합을 피한다. 현재 머신에는 해당 모델·MLX 런타임·서버가 설치돼 있지 않으므로 tok/s 수치는 아직 없다. 이름만 보고 성능을 단정하지 말고 실제 야구 영상 벤치마크로 lane을 확정해야 한다.

## 왜 이 두 모델인가

### Gemma 4 12B Unified — 빠른 비전/오디오 lane

- 2026년 7월 공개 기술 보고서(`arxiv:2607.02770`)가 연결된 공식 모델이다.
- 12B Unified 모델 카드에는 256K context와 text/image/audio 입력이 명시되고, 제품 설명에는 video 입력도 명시된다.
- Google이 QAT Q4_0 GGUF를 직접 제공하며 llama.cpp 실행법을 공식 카드에 제공한다. 현재 Mac에는 Metal 지원 llama.cpp build 8680이 이미 있다.
- 6.98GB 가중치는 반복적인 품질 검사·키프레임 해석에 유리하다.
- 단, 실제 llama.cpp 빌드의 **video/audio API 지원 범위는 이미지 지원과 별개**이므로 설치 후 계약 테스트가 필요하다. 지원이 모자라면 원본 Transformers 런타임 또는 별도 미디어 전처리 어댑터를 쓴다.

### Qwen3.6-35B-A3B — 강한 종합 추론 lane

- 공식 모델 카드는 총 35B 중 3B 활성 MoE, vision encoder, 262K native context(YaRN으로 최대 1.01M), 장시간 video 전처리 가이드를 명시한다.
- 공식 카드의 자체 보고치에는 video understanding 및 agent/coding 벤치마크가 포함돼 있어 구조화 도구 호출과 증거 종합에 적합한 후보이다.
- MLX Community의 변환 카드는 `mlx-vlm 0.4.4`, 4-bit 약 20.4GB, 이미지 입력 예제를 명시한다. M5 Max 128GB에서 용량상 여유가 크다.
- 4-bit 양자화가 공간 관계·미세 차이 판정에 미치는 영향은 야구 전용 검증이 없다. 따라서 픽셀 좌표·각도 계산은 절대 맡기지 않고, 원본 프레임과 결정론적 수치의 일관성 검토에만 쓴다.

### 제외/보류 원칙

- 2026년 8월 이후 업로드, 비공식 `abliterated/uncensored` 파생본, 출처·라이선스가 불명확한 모델은 2026-07 기준선에서 제외한다.
- GLM-4.6V-Flash 등 더 오래된 후보는 위 두 모델의 실제 벤치마크가 실패할 때만 fallback으로 비교한다.
- 어떤 모델도 “최신”이라는 이유만으로 채택하지 않는다. 한국어 코칭 정확성, frame ordering, 좌우 반전, 시간적 인과, JSON schema 준수, hallucination rate를 같은 골든셋에서 측정한다.

## VLM에 줄 입력과 주지 말아야 할 권한

VLM 입력은 전체 영상을 무작정 토큰화하지 말고 다음 **Evidence Packet**으로 고정한다.

```json
{
  "run_id": "...",
  "athlete_profile": {"bats": "R", "age_band": "HS"},
  "capture_quality": {"fps": 240, "view": "side", "score": 0.91},
  "events": [{"name": "contact", "t_ms": 842, "confidence": 0.76}],
  "metrics": [{"name": "hip_shoulder_separation", "value": 31.2, "unit": "deg", "ci95": [27.8, 34.9]}],
  "keyframes": ["address", "foot_plant", "launch", "contact", "finish"],
  "overlays": ["pose", "bat_axis", "uncertainty"],
  "quality_flags": ["contact_blur"]
}
```

허용 역할:

- 촬영 각도/가림/blur/잘못된 타자 선택의 semantic QA
- 프레임과 수치가 서로 모순되는지 검토
- 같은 선수의 여러 스윙 비교와 변화 요약
- 허용된 근거만 인용한 한국어 코칭 문장 생성
- 신뢰도 낮을 때 재촬영 또는 사람 검토 요청

금지 역할:

- 이미지에서 직접 bat speed, exit velocity, 3D attack angle을 숫자로 추측
- ground truth 없이 “정확/이상적/부상 위험”을 단정
- MEDA의 기준값을 매번 자유 생성
- CV confidence가 낮은데도 피드백을 강행

MEDA 기준은 버전 관리된 지식 베이스/룰로 먼저 고정하고, APA는 구조화 evidence만 읽으며, APE는 인용 가능한 `evidence_id` 없이 문장을 생성하지 못하게 한다. 모든 출력은 JSON Schema로 검증한다.

## 비전을 최대한 활용하는 목표 파이프라인

### 0. 촬영 품질 게이트

분석 전에 해상도, 실제 timestamp/FPS, 셔터 blur, 카메라 이동, 전신 가시성, 타석 방향, 정면/측면 각도, 가림, 조명, 배경을 채점한다. 점수가 기준 미달이면 비싼 추론 대신 재촬영 지침을 즉시 준다.

권장 촬영:

- 자세/회전: 고정 삼각대, 1080p 이상, 가능하면 정면과 측면 2대.
- 배트/공/접촉: 120fps는 최소, 240fps 권장, 빠른 셔터와 충분한 조명, 기준 길이 또는 카메라 캘리브레이션 포함.
- 원본 timestamp를 보존하고 VFR 영상을 CFR로 바꿀 때 frame index가 아니라 시간 매핑표를 남긴다.

### 1. 사람·포즈·실루엣

- 기존 YOLO26m-pose를 유지하되 COCO17만으로는 손·발·골반·흉곽 표현이 부족하므로 baseball-specific keypoint 또는 whole-body schema를 추가한다.
- 공식 YOLO26은 pose 외에 detection, instance/semantic segmentation, depth, OBB를 같은 계열로 지원한다. 선수 mask로 배경과 다른 사람을 제거하고, OBB/custom keypoint로 bat knob/tip과 bat axis를 학습하는 것이 일반 COCO `baseball bat` box보다 낫다.
- ByteTrack ID 하나만 믿지 말고 유니폼/타석 ROI/시간적 keypoint 품질을 함께 사용해 batter identity score를 만든다.
- keypoint마다 좌표뿐 아니라 confidence/covariance, interpolation 여부, 관측/추정 상태를 보존한다.

### 2. 배트·공·손의 고속 추적

- 배트는 axis가 중요한 길쭉한 물체다. 단순 axis-aligned box 대신 `(knob, tip)` keypoint + OBB/segmentation을 조합한다.
- 3-frame TrackNet만 고정하지 말고 optical flow/temporal heatmap/pose-conditioned ROI를 동일 데이터에서 ablation한다.
- 손목 주변 ROI로 탐색 범위를 줄이되 wrist proxy를 관측값과 섞지 않는다. `observed`, `interpolated`, `proxy`를 분리한다.
- 공은 별도 small-object detector/temporal tracker가 필요하다. 공이 없으면 exit velocity·launch angle은 출력하지 않는다.
- 이벤트 주변은 원본 해상도·고 FPS를 유지하고, 일반 구간만 downsample한다.

### 3. 이벤트 검출

현재의 “손 속도 최대 = contact”를 폐기하고 address → load → stride → foot plant → launch → contact → extension → finish의 상태 모델로 바꾼다. 입력은 몸 keypoint, bat axis/속도, 공 궤적, audio impact transient이며 각 이벤트에 timestamp와 confidence interval을 준다. 사람 라벨 이벤트를 최소 200~500 스윙 확보해 frame/timestamp MAE를 측정한다.

### 4. 3D와 캘리브레이션

- MotionBERT는 adapter 뒤에 유지하되 모델 교체 가능하게 한다. MotionBERT, 최신 temporal pose/mesh 후보, 2-view triangulation을 동일 H36M/SMPL 중간 표현으로 비교한다.
- monocular 3D는 절대 scale과 depth가 모호하다. 선수 키 하나로 전체 3D를 “정답”으로 만들지 말고, camera calibration/ground plane/두 번째 시점이 있을 때만 metric 3D를 표방한다.
- camera-space, world-space, image-space를 타입과 필드명으로 분리하고 좌/우타자 mirror transform을 한 곳에서만 수행한다.
- temporal smoothing 뒤 원신호도 보존하고 지연/peak attenuation을 측정한다.

### 5. 지표와 불확실성

모든 지표에 `value`, `unit`, `space`, `event`, `confidence`, `CI`, `source joints/frames`, `definition_version`을 붙인다. angle은 wrap/부호/축 정의를 테스트하고, bat speed는 scale + timestamp + bat 관측률이 모두 통과할 때만, exit velocity는 공 관측과 calibration이 있을 때만 제공한다.

### 6. 로컬 VLM 검토와 피드백

fast lane이 각 스윙의 키프레임·품질 flag·수치 모순을 먼저 검사한다. low-confidence, 좌우 혼동, 모델 간 불일치, 장기 비교 요청만 strong lane으로 올린다. 최종 보고서는 각 주장에 영상 timestamp와 metric ID를 링크하고, 확신도와 재촬영 필요 여부를 표시한다.

## 물리적으로 넘을 수 없는 한계

- 30fps는 프레임 간격이 33.3ms다. 70~90mph(약 31~40m/s) 배트는 한 프레임 사이 약 1.0~1.34m 이동하므로 contact 순간과 peak speed를 신뢰성 있게 복원할 수 없다.
- 120fps에서도 약 0.26~0.34m/프레임, 240fps에서도 약 0.13~0.17m/프레임이다. 짧은 노출, sub-frame fit, scale calibration이 필요하다.
- 단일 2D 영상의 wrist path attack angle은 bat barrel의 3D attack angle이 아니다. 현재 문서의 “exact attack angle” 표현은 image-plane 추정치로 바꿔야 한다.
- exit velocity는 배트만 추적해서 얻을 수 없다. 공의 contact 전후 궤적, 시간, scale이 필요하다.
- 로컬 VLM을 추가해도 이 관측 한계는 사라지지 않는다. 모델은 누락된 물리를 그럴듯하게 채울 뿐이다.

## 실측 벤치마크 설계

모델/런타임 설치 후 다음 네 부하를 cold/warm 각각 5회 측정한다.

| 부하 | 입력 | 품질 지표 | 운영 지표 |
|---|---|---|---|
| short QA | 5 keyframes + 촬영 metadata | 불량 촬영 분류 F1 | TTFT, peak RSS |
| approval | confidence/flag JSON | 분석/재촬영/사람검토 정확도 | p50/p95 latency |
| visual reasoning | event 전후 16~32 frames + overlay | 좌우/순서/모순 오류율 | tokens/s, media preprocess 시간 |
| report | 3스윙 evidence packet | 근거 인용률, hallucination, 한국어 코치 rubric | 총시간, peak RSS, schema retry율 |

fast 모델이 strong 모델보다 느리거나 품질 gate를 자주 실패하면 lane을 교체한다. 256K/1M context는 기본값이 아니다. 이 앱은 필요한 event window만 보내 16K~32K 내에서 시작하고, 장기 시즌 비교도 수치 요약 + 대표 근거로 압축한다.

## 단계별 구현 우선순위

### 0단계 — 측정 가능한 기반 (가장 먼저)

- 경로/환경/lockfile/run manifest 정리
- 촬영 프로토콜과 20~50개 골든 스윙
- timestamp·좌표계·metric schema와 테스트
- 현재 2D/3D 지표의 ground-truth 비교 및 위험한 문구 제거

### 1단계 — 신뢰 가능한 MVP

- 촬영 품질 gate, 타자 선택, COCO17 품질/결측 처리
- 이벤트 상태 모델과 신뢰도
- Evidence Packet + Gemma fast lane 리포트
- UI에서 원본/overlay/metric 근거를 같은 timestamp로 재생

### 2단계 — 핵심 차별화

- 240fps bat knob/tip + OBB/segmentation 학습
- 공 추적, 카메라/scale calibration
- Qwen strong lane의 다중 스윙 비교와 모순 검토
- 모델별/타자별/시점별 회귀 대시보드

### 3단계 — 고급 정확도

- 2-view triangulation 또는 검증된 3D mesh/pose ensemble
- bat speed/attack angle/exit metrics의 계측기 비교
- 개인 baseline과 시간에 따른 변화, 코치 검토 workflow

## 채택 기준

기능 수가 아니라 다음 gate로 출시를 판단한다.

- 사람/타자: track ID 안정성, keypoint PCK/누락률.
- 배트/공: contact-zone recall, knob/tip error, trajectory continuity.
- 이벤트: contact/foot-plant timestamp MAE와 coverage.
- 3D: MPJPE/각도 MAE 및 view별 편향.
- 지표: 계측기 대비 MAE/ICC, 좌우타자/시점별 calibration.
- 피드백: 근거 인용 100%, 금지 주장 0건, 코치 rubric와 inter-rater agreement.
- 운영: 골든셋 회귀 통과, 실패 시 abstain, 완전 로컬 실행에서 원본 영상 외부 전송 0건.

## 확인한 공식 자료

- [Qwen3.6-35B-A3B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6 MLX 4-bit 변환 카드](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)
- [Gemma 4 12B Unified 공식 모델 카드](https://huggingface.co/google/gemma-4-12B-it)
- [Gemma 4 12B 공식 QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
- [Ultralytics YOLO26 공식 문서](https://docs.ultralytics.com/models/yolo26/)
- [MotionBERT 공식 저장소](https://github.com/Walter0807/MotionBERT)

공식 카드의 vendor benchmark는 후보 선정 근거일 뿐 독립 검증이 아니다. 최종 모델 선택은 이 프로젝트의 골든셋 실측으로 확정해야 한다.
