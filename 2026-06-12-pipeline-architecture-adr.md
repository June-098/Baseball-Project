# ADR-001: Baseball Video Analysis — Pipeline Code Architecture

**Status:** Proposed
**Date:** 2026-06-12
**Deciders:** June

## Context

Work so far has been a single, growing Colab notebook for Phase 1 (pose estimation):
mount Drive → load videos → YOLO26-pose + ByteTrack → export keypoints CSV → select the batter's track.

The roadmap has six phases (pose validation, bat tracking, 3D lifting, event detection, metrics, product), each producing output the next phase consumes. The notebook approach works for today's task but doesn't scale as more phases are added — and eventually this same logic needs to run inside a FastAPI backend, not just Colab.

**Constraints:** solo developer, prototyping in Colab with GPU, no production traffic yet, video storage/infra deliberately deferred.

## Decision

Adopt a **modular pipeline package**: each phase becomes a small Python module with a clear input/output contract (CSV/JSON files). Colab notebooks become thin wrappers that call these modules. This lets prototyping continue in Colab while the same code becomes directly reusable by the future FastAPI backend — no rewrite later.

## Proposed Codebase Structure

```
baseball-cv/
├── pipeline/
│   ├── config.py              # KEYPOINT_NAMES, model paths, constants
│   ├── io_utils.py             # video/CSV/JSON read-write helpers
│   ├── pose/
│   │   ├── extractor.py        # YOLO26-pose + ByteTrack → keypoints_raw.csv
│   │   └── batter_selector.py  # track selection (persistence + size heuristic)
│   ├── bat/
│   │   ├── detector.py         # bat_detector_v1.pt inference
│   │   └── association.py      # link bat bbox to batter's wrist keypoints
│   ├── lifting/
│   │   └── motionbert_runner.py
│   ├── events/
│   │   └── event_detector.py   # swing phase classifier
│   ├── metrics/
│   │   └── metrics.py          # attack angle, bat speed, swing path, etc.
│   └── orchestrator.py         # runs all phases end-to-end on one video
├── notebooks/
│   ├── 01_pose_validation.ipynb
│   ├── 02_bat_detector_training.ipynb
│   └── ...
├── data/
│   ├── raw_videos/              # Drive-mounted in Colab
│   ├── keypoints/
│   ├── annotations/             # CVAT exports
│   └── models/                  # .pt weights
└── tests/
    └── test_pose_extractor.py
```

## Stage Contracts (Data Flow)

Keeping each stage's input/output as a flat file means stages stay decoupled, testable in isolation, and swappable for database storage later without touching the logic.

| Stage | Input | Output |
|---|---|---|
| `pose.extractor` | video | `keypoints_raw.csv` (video, frame, track_id, bbox, keypoint, x, y, confidence) |
| `pose.batter_selector` | `keypoints_raw.csv` | `keypoints_batter.csv` (filtered to one track_id) |
| `bat.detector` | video | `bat_boxes.csv` (video, frame, bbox) |
| `bat.association` | `keypoints_batter.csv` + `bat_boxes.csv` | `batter_with_bat.csv` |
| `lifting.motionbert_runner` | `keypoints_batter.csv` | `keypoints_3d.json` |
| `events.event_detector` | `keypoints_3d.json` | `events.json` (frame indices per swing phase) |
| `metrics.*` | `keypoints_3d.json` + `events.json` + `batter_with_bat.csv` | `metrics.json` |

## Options Considered

### Option A: Single growing notebook
| Dimension | Assessment |
|---|---|
| Complexity | Low now, very high by Phase 3-4 |
| Cost | None |
| Scalability | Poor — logic can't be reused by a backend |
| Team familiarity | High (current approach) |

**Pros:** fastest right now, zero setup.
**Cons:** unmaintainable once 4-5 phases are chained together; FastAPI backend would need a full rewrite; hard to test stages independently.

### Option B: Modular pipeline package (recommended)
| Dimension | Assessment |
|---|---|
| Complexity | Medium — small upfront refactor |
| Cost | None |
| Scalability | High — same modules run in Colab, FastAPI, or batch jobs |
| Team familiarity | Medium |

**Pros:** notebooks become thin callers; FastAPI imports the same modules later; each stage independently testable; maps 1:1 to the 6-phase roadmap.
**Cons:** requires a small refactor now, while still mid-Phase-1.

### Option C: Per-stage microservices (one container per phase)
| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | Higher — multiple deployments |
| Scalability | High, but premature |
| Team familiarity | Low |

**Pros:** ultimate independent scalability and deploys.
**Cons:** massive overcomplexity for a solo developer with no production traffic — premature optimization.

## Trade-off Analysis

Option A is fine for the next hour but becomes a liability the moment Phase 2 starts — bat detection needs to reuse the batter-selection logic from Phase 1, and a notebook makes that copy-paste rather than reuse. Option C solves a scaling problem that doesn't exist yet (one user: you). Option B is the right size: a small refactor now means every future phase is "add a module" rather than "untangle a notebook."

## Consequences

- **Easier:** testing each stage independently, reusing pose/bat logic in the future FastAPI backend, handing off code to a collaborator later.
- **Harder:** a bit more setup right now — creating the package skeleton and config file before continuing Step 4/5.
- **Revisit later:** once the FastAPI backend exists, decide whether the pipeline runs synchronously per request or via a job queue (Celery/SQS). That's a Phase 6+ decision, not now.

## Action Items

1. [ ] Create the `baseball-cv/` package skeleton (`pipeline/`, `notebooks/`, `data/`, `tests/`)
2. [ ] Move current Step 1-5 code into `pipeline/pose/extractor.py` and `pipeline/pose/batter_selector.py`; move `KEYPOINT_NAMES` and paths into `config.py`
3. [ ] Rewrite the Colab notebook as a thin runner: `from pipeline.pose import extractor, batter_selector`
4. [ ] Lock down the CSV/JSON schema for each future stage now, before building it
5. [ ] Reference this ADR from MEMORY.md under Key Decisions
