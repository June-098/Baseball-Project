# Personal Baseball Project 감사 인덱스

## 먼저 읽을 문서

- [종합 감사 보고서](</Users/mac/Personal Baseball Project-audit/reports/00-EXECUTIVE-REPORT.md>) — 최종 판정, 상위 결함, Go/No-Go, 목표 아키텍처, 2026-07 모델 선택, 구현 순서.

## 클러스터 보고서

| 보고서 | 담당 범위 | 핵심 질문 |
|---|---|---|
| [01 핵심 비전](</Users/mac/Personal Baseball Project-audit/reports/01-core-vision.md>) | `run_pipeline.py`, `config.py`, `src/*`, batter selector, README/ADR | 전체 호출 그래프·데이터 계약·포즈/3D/metric 버그는 무엇인가? |
| [02 추적·실험](</Users/mac/Personal Baseball Project-audit/reports/02-tracking-experiments.md>) | 날짜형 실험, Colab, 노트북, 가중치, TrackNet/Kalman/MotionBERT | 배트·공 추적 실험이 제품 코드와 연결되고 검증됐는가? |
| [03 도메인·제품](</Users/mac/Personal Baseball Project-audit/reports/03-domain-product.md>) | feedback engine, Savant 문서, 지표, UX, 데이터·privacy·safety | 수치와 코칭 주장이 야구/생체역학·제품 수준에서 타당한가? |
| [04 엔지니어링·런타임](</Users/mac/Personal Baseball Project-audit/reports/04-engineering-runtime.md>) | 빌드·의존성·보안·테스트·Mac·현재 모델 lane | 이 ZIP을 재현 가능하고 안전한 로컬 앱으로 만들 수 있는가? |
| [05 비전·로컬 LLM 로드맵](</Users/mac/Personal Baseball Project-audit/reports/05-vision-local-llm-roadmap.md>) | YOLO26 확장, 고속 추적, evidence packet, Gemma/Qwen | 2026-07 기준 어떤 비전/로컬 VLM 구조로 재설계할 것인가? |

## 감사 산출물 위치

- 분석용 소스 복사본: `/Users/mac/Personal Baseball Project-audit/source/Personal Baseball Project`
- 보고서 폴더: `/Users/mac/Personal Baseball Project-audit/reports`
- 원본 ZIP: `/Users/mac/Downloads/Personal Baseball Project.zip` — 변경 없음.
