# Baseball Batting Analysis — CV Pipeline

A single-camera (phone video) pipeline that turns batting footage into body-mechanics
data and feedback. You upload a swing clip; the pipeline finds the batter, tracks their
body, lifts the pose into 3D, and draws the swing-relevant angles back onto the video.

This repo is the code version of the research notebook `YOLO_Baseball.ipynb`. It contains
the validated parts of that notebook only — pose extraction, batter selection, MotionBERT
3D lifting, and the 2D/3D visualizations. **Bat tracking is deliberately left out** (see
*What's not here* below).

> Plain-language note: every metric below lists its unit and what "good" looks like for a
> high-school athlete. Technical terms are defined on first use.

---

## The pipeline, stage by stage

| Stage | Module | Input → Output | What it does |
|---|---|---|---|
| 1. Pose | `src/pose_extraction.py` | videos → `keypoints_raw.csv` | Runs YOLO26 pose ("pose" = finding 17 body points per person) + ByteTrack (gives each person a stable ID) on every frame. |
| 2. Batter | `src/batter_selection.py` | `keypoints_raw.csv` → `keypoints_batter.csv` | Keeps only the batter (the largest person in frame) and splits a multi-swing clip into numbered `segment_id`s. |
| 3. 3D lift | `src/motionbert_*.py` | `keypoints_batter.csv` → `keypoints_3d.json` | MotionBERT turns the flat 2D skeleton into 3D (x, y, z per joint). "Lifting" = inferring depth from a single camera. |
| 3b. 3D viz | `src/visualize_3d.py` | → `skeleton_3d_*.mp4` | Animated 3D skeleton, one per swing, for visual QA. |
| 4. 2D viz | `src/visualize_2d.py` | → `skeleton_2d_*.mp4` | Plain 15-dot skeleton drawn on the original video. |
| 5. **2D domain** | `src/apply_2d_domain.py` | → `skeleton_2d_biomech_*.mp4` + `2d_metrics.json` | **The analysis overlay.** Draws the swing angles that matter and reports them as live numbers (see below). |
| 6. **3D metrics** | `src/compute_3d_metrics.py` | `keypoints_3d.json` → `metrics_3d.json` | **View-independent** versions of the swing angles (true hip-shoulder separation, lead-arm extension, etc.) — removes the 2D "proxy" caveat. |

### Run it

```bash
python run_pipeline.py                  # full pipeline
python run_pipeline.py --stage apply2d  # just the 2D-domain overlay (stage 5)
python run_pipeline.py --stage metrics3d # just the 3D metrics (stage 6)
python src/apply_2d_domain.py --videos "Chae_friend_Righty_Batting_V1.mov"   # one clip
```

> **Torch note:** `pose`/`motionbert` need ultralytics/torch (GPU in Colab). The analysis
> stages (`batter`, `viz2d`, `apply2d`, `metrics3d`) are torch-free and run anywhere.

Paths live in `config.py`. Google Drive is mounted locally at `G:/My Drive` (this is the
same folder as Colab's `/content/drive/My Drive`), so outputs land in
`G:/My Drive/Baseball Project/`.

---

## Stage 5: the 2D-domain overlay (the headline feature)

The plain skeleton (stage 4) is pretty but not useful. Stage 5 is the agent team's answer
to *"how do we apply the 2D domain effectively?"* — it draws the four swing angles a coach
actually reads, straight from the 2D body points (no bat tracking, no 3D needed):

| Metric (unit) | What it means | Good for a HS hitter |
|---|---|---|
| **Hip-shoulder separation** (°) | How much the shoulders out-rotate the hips — the "wind-up" that stores power. | ~25–45° at launch |
| **Spine tilt** (°) | How far the torso leans from vertical. | Consistent, not drifting |
| **Front-knee angle** (°) | Interior angle of the lead knee (180° = straight). A firm front leg transfers energy up. | Braces toward ~150–170° at contact |
| **Attack angle** (°) | Direction the hands are travelling at contact (the body-keypoint proxy for MLB attack angle). Positive = swinging up. | **5–20°** (the MLB "ideal" band) |

**Honest caveat:** these are measured in the flat image plane, so they are *view-dependent
proxies*, not lab-grade 3D angles. That's the same "approximable from body keypoints" tier
Baseball Savant describes. The exact versions need the 3D pose (MotionBERT) and, for true
attack angle/bat speed, bat tracking (deferred).

The "contact" frame is estimated as the moment the hands move fastest — a lightweight
stand-in until real event detection (Phase 4) lands.

The agent reasoning behind this design lives in `feedback-engine/` (MEDA → APA → APE).

---

## Colab cells (`colab/`)

The work that needs a GPU or that you run interactively lives as self-contained Colab cells
(also copied to `Drive/colab/`):

| Cell | Does |
|---|---|
| `pose3d_estimation_colab.py` | MotionBERT 2D→3D lifting + animated 3D skeleton (clean cells 14–18). |
| `pose3d_metrics_colab.py` | The 3D biomech metrics (= `src/compute_3d_metrics.py`). |
| `apply_2d_domain_colab.py` | The 2D-domain overlay (= `src/apply_2d_domain.py`). |
| `tracknet_bat_tracking_colab.py` | **Bat tracking** — TrackNet-style temporal model: label prep, train, infer. |
| `bat_attack_angle_colab.py` | **Exact** attack angle from the tracked bat sweet spot. |

---

## Bat tracking (in progress — TrackNet)

Pretrained YOLO collapses to ~17% detection in the contact zone (the bat is a motion blur at
70–90 mph on a 30fps phone), so the bat-detection QA and the Kalman/wrist-proxy experiment are
left out of the repo. The replacement is `colab/tracknet_bat_tracking_colab.py`: a temporal
model that ingests **3 frames at once** so motion helps instead of hurts, tracks the **knob +
tip** (→ orientation + sweet spot), and uses **motion-blur augmentation** to survive the contact
zone. It is **not zero-shot** — it needs ~200–500 labeled frames *including contact-zone frames*
(bootstrap from YOLO boxes + label the blurred frames by hand, e.g. CVAT). Once it tracks the
sweet spot, `bat_attack_angle_colab.py` gives the **exact** attack angle (the 2D overlay's
wrist-path number was a proxy for exactly this).

Still deferred: bat **speed** and swing length (need calibrated scale), and the React Native app.

---

## Layout

```
config.py              paths + joint/skeleton constants
run_pipeline.py        orchestrator (--stage ...)
src/                   one module per pipeline stage
colab/                 self-contained Colab cells (GPU / interactive work)
feedback-engine/       MEDA/APA/APE agent artifacts (one file per agent per run)
models/MotionBERT/     checkpoint location (falls back to G:/My Drive/models/MotionBERT)
```
