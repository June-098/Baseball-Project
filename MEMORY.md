# Personal Baseball Project Memory

*Last updated: 2026-09-01*

---

## Working Preferences

- **Summaries** (e.g. posted to Slack, end-of-session recaps): keep under 500 words.

---

## Active Projects

- **Baseball Video Analysis App** — Building a mobile + web app where athletes upload batting/pitching video and receive AI-driven mechanics feedback. Phase 1 = batting only, high school athletes. Phase 2 = pitching.

---

## Contacts

*Add coaches, advisors, or technical collaborators here as they come up.*

---

## Key Decisions

### 2026-06-09 — Tech Stack Locked

**Problem:** Needed to select the right CV/ML stack for single-camera video analysis (phone uploads) without requiring specialized hardware.

**Decision:** Rejected ByteTrack as the core tool (it only produces bounding boxes, not joint keypoints). Rejected Theia3D for consumer use (requires multi-camera lab rig). Selected the following pipeline:

| Layer | Tool | Rationale |
|---|---|---|
| Pose estimation (server) | YOLOv8-Pose | Best accuracy on uploaded video |
| Pose estimation (on-device) | MediaPipe Pose | Runs on-device, 33 landmarks, real-time |
| 2D → 3D Lifting | MotionBERT | SOTA single-camera 3D, same approach as SportFX |
| Ball tracking | TrackNet | Purpose-built for fast sports balls; standard YOLO fails on 90mph pitches |
| Event detection | Custom LSTM/Transformer | Detects swing phases: toe touch, heel plant, launch, contact, extension |
| Annotation | CVAT | Label baseball-specific training footage |
| Backend | FastAPI (Python) | Async, ML-friendly |
| GPU inference | AWS EC2 g4dn or Modal.com | Cost-effective on-demand GPU |
| App | React Native | One codebase → iOS + Android + Web |
| Storage | AWS S3 + CloudFront | Standard scalable video storage |
| Training data | Statcast + YouTube pro footage | Ground truth for fine-tuning models |

**Why ByteTrack stays in the file:** Potentially useful as a pre-processing step to crop/isolate the player from a multi-person frame. Not analysis.

**Why Theia3D stays in the file:** Useful as benchmark reference and future premium hardware tier inspiration. Not consumer-shippable.

---

### 2026-06-09 — MVP Scope

**Decision:** Phase 1 = batting mechanics only, targeting high school athletes. Pitching is a separate biomechanical problem; solving both simultaneously will stall development.

**Phase 1 target metrics (batting):**
- Attack angle
- Hip rotation timing
- Hip-shoulder separation
- Bat speed (estimated from joint velocities)
- Extension at contact
- Swing path (upper/lower cut, flat)
- Time to contact

**Phase 2 (pitching):** Arm path, hip-shoulder separation, release point consistency, pitch spin proxy.

---

### 2026-09-01 — SwingLens chat is report-aware

**Decision:** After a clip is analyzed, the hitting-coach chat receives that swing’s prototype score and checkpoint cues and must answer “why did I score X” from those numbers. Off-topic refusals stay 1–2 sentences with no drill attached.

---

### 2026-09-01 — On-device pose is IMAGE mode plus square-pad

**Decision:** Overlay and impact both failed under VIDEO-mode tracking on seeked frames (stale skeleton, peak wrist speed after contact). Pose now runs IMAGE mode on a square-padded frame mapped back to the source, and impact is the last high-speed sample with the hands still below the head. Fine-tuning a baseball pose model is the next step only if Lite still misses after this mapping.

---

## Competitive Landscape

| Tool | Approach | Weakness | Our Edge |
|---|---|---|---|
| Hawk-Eye | Multi-camera stadium install | Not consumer-accessible | We work from phone video |
| HitTrax | Radar + camera, facility-based | Hardware dependent | No hardware needed |
| Rapsodo | Portable radar | Hardware dependent | No hardware needed |
| DiamondKinetics | Bat-mounted IMU wearable | No vision data, no ball flight | We capture full body + ball |
| SportFX | Monocular 3D lifting, general sports | Not baseball-specific | Baseball-specific model fine-tuning + high school focus |

---

## Research Log

### 2026-06-09
- Studied Theia3D markerless motion capture: gold standard multi-camera system, 124 landmarks, bat + ball tracking. Not viable for consumer app but valuable as benchmark.
- Studied ByteTrack MOT: multi-object bounding box tracker. Not the right core tool; limited to player isolation pre-processing at most.
- Confirmed SportFX uses monocular 3D lifting (2D video → 3D pose). Same approach is viable for us with MotionBERT.
- Identified key swing events to detect: toe touch, heel plant, swing launch, contact, extension.
- Researched skeleton body labeling: 17 keypoints (head, torso, center hip, L/R shoulder, elbow, hand, hip-joint, knee, foot).

### Next Research Items
- [ ] MotionBERT: does it have existing sports/baseball fine-tuning examples?
- [ ] TrackNet: what adaptations are needed for baseball vs. its original badminton use case?
- [ ] Training data strategy: how to source and annotate enough pro player batting clips for fine-tuning
- [ ] Truth source decision: Driveline partnership vs. coach advisory board vs. Statcast metric alignment

---

### 2026-06-11 — Pose Model Updated to YOLO26, Implementation Roadmap Set

**Decision:** Use `yolo26m-pose.pt` (released Jan 2026, NMS-free, edge-first, COCO-pretrained) instead of YOLOv8-Pose. Same role in the stack — supersedes the earlier choice.

**Key constraints/clarifications from this session:**
- Run on **GPU (T4 in Colab)**, not TPU — YOLO26 is PyTorch-based; TPU requires PyTorch/XLA conversion, not worth it.
- No training-from-scratch needed. Pretrained model already detects 17 COCO keypoints on the batter.
- **Multi-person handling is pipeline code, not a model problem** — select largest bounding box (closest to camera) as the batter.
- **Bat tracking is a separate small object detector**, not a fine-tune of the pose model. Fine-tune `yolo26n.pt` (detection) on ~50-100 CVAT-annotated bat bounding boxes → produces a new `bat_detector_v1.pt`. Run alongside the pose model and combine outputs in pipeline code.
- **Fine-tune the pose model only if systematic failures appear** across multiple test clips (e.g., wrist occlusion by bat during every swing) — one-off glitches get handled by smoothing/interpolation later, not retraining.
- No public baseball-specific pose keypoint dataset exists. MLB-YouTube dataset (GitHub: piergiaj/mlb-youtube) is a raw video source only (labeled for activity recognition, not keypoints) — useful for generating pseudo-labels via the pretrained model + manual CVAT correction if fine-tuning becomes necessary.

---

## Pipeline Build Order (Updated 2026-06-12)

Each step lists **Implement** (what to build/run), **Verify** (how to confirm it ran correctly), and **Analyze** (what to look at in the output to decide next action).

**Phase 1 — Pose Estimation Validation (✅ Complete)**
1. **Implement:** Colab + GPU (T4), load `yolo26m-pose.pt`, mount Drive, run `model.track(persist=True, tracker="bytetrack.yaml")` on test clips → export `keypoints_raw.csv` (video, frame, track_id, bbox, keypoint, x, y, confidence).
   **Verify:** row count per video matches frame_count × num_detected_people × 17 keypoints; no missing videos.
   **Analyze:** done — found multi-swing source files produce many track_ids per video (e.g., V2 had 18).
2. **Implement:** `pipeline/pose/batter_selector.py` — `select_batter()` picks the largest-bbox track per frame, `summarize_segments()` groups consecutive frames into `segment_id` → export `keypoints_batter.csv`.
   **Verify:** run `summarize_segments()`; segment count should match the number of distinct swings you recorded in that file.
   **Analyze:** done — validated 100% accurate on V2 (6 segments, 444 frames, up to 3 people/frame). Confidence > 0.9 rejected as a selection signal (non-batter tracks scored higher).
3. **Status:** `keypoints_batter.csv` (per-segment, batter-only, 17 keypoints) is the finalized Phase 1 output and the input to Phase 3.

**Phase 2 — Bat Detection (⏸ DEFERRED to after Phase 3)**
- QA of pretrained YOLO bat detection already done (2026-06-17): detection rate 60–75% overall, drops to 17–18% in the contact zone for single-swing clips. Gaps of 10–13 consecutive frames at peak swing. YOLO confidence 0.4–0.6 — unreliable at the critical moment.
- Deferred decision: bat tracking (Phase 2 + 2.5) will resume after Phase 3 body lifting is working. Recommended approach when returning: TrackNet-style temporal model or SAM2 (zero-shot propagation), NOT more YOLO fine-tuning.

**Phase 3 — 2D → 3D Body Lifting (🔄 IN PROGRESS — current focus)**
8. **Implement:** `2026-06-17-phase3-motionbert.py` — feeds `keypoints_batter.csv` into MotionBERT (DSTformer, MB_ft_h36m checkpoint, H36M fine-tuned, 37.2mm MPJPE) → `keypoints_3d.json`. Checkpoint saved at `My Drive/models/MotionBERT/best_epoch.bin`.
   **Verify:** no NaNs/gaps in 3D coordinates across all frames of a segment.
   **Analyze:** sanity-check joint angle ranges (elbow flexion, hip/shoulder rotation) fall within plausible human limits for each segment. Cell F renders animated 3D skeleton MP4 per segment for visual QA.

**Next session starts at:** Running `2026-06-17-phase3-motionbert.py` in Colab — download MB_ft_h36m checkpoint first.

**Phase 4 — Event Detection**
9. **Implement:** classifier (LSTM/Transformer) on the 3D joint time series (+ bat path from Phase 2.5, once available) → `events.json` with frame indices for toe touch, heel plant, launch, **contact**, extension.
   **Verify:** overlay predicted event frames on the source video for a few segments; manually confirm contact frame aligns with visible bat-ball contact.
   **Analyze:** check event timing consistency across segments of the same swing type — large variance flags either bad 3D data or a model issue.

**Phase 5 — Metrics**
10. **Implement:** compute attack angle, hip-shoulder separation, bat speed, swing path, extension, hip rotation timing, time to contact using `keypoints_3d.json` + `events.json` + bat path → `metrics.json`.
    **Verify:** spot-check 1-2 segments by hand (e.g., manually measure attack angle from a frame) and compare to computed value.
    **Analyze:** validate against truth source (still open — see Next Research Items).

**Phase 6 — Product**
11. Feedback + drill mapping engine.
12. Progress tracking across sessions.
13. React Native app (after pipeline proven).

**Deferred:** Video upload + S3 storage + privacy handling — explicitly pushed to after pipeline validation (2026-06-11 decision).
**Deferred:** Bat tracking (Phase 2 + 2.5) — pushed to after Phase 3. YOLO bat detection proven insufficient (17% in contact zone). Resume with TrackNet or SAM2.

---

### 2026-06-12 — Codebase Architecture: Modular Pipeline Package

**Decision:** ADR-001 (see `2026-06-12-pipeline-architecture-adr.md`) — restructure Phase 1-5 code as a modular `baseball-cv/` package (`pipeline/pose/`, `pipeline/bat/`, `pipeline/lifting/`, `pipeline/events/`, `pipeline/metrics/`) with flat-file (CSV/JSON) contracts between stages. Colab notebooks become thin wrappers calling these modules. Chosen over a single growing notebook (unmaintainable past Phase 2) and per-stage microservices (premature for a solo dev with no production traffic). This same code becomes directly reusable by the future FastAPI backend.

**Action before continuing Phase 1 Step 4/5:** create the package skeleton and move existing pose extraction + batter selection code into `pipeline/pose/extractor.py` and `pipeline/pose/batter_selector.py`.

---

### 2026-06-12 — Bat Detection Moved Up to Phase 1 (uses pretrained COCO class)

**Decision:** COCO (which `yolo26m.pt` is pretrained on) includes a "baseball bat" class — pretrained bat detection works with zero training. Combined former Steps 4 and 5 into one pass:

- **Step 4:** Run `yolo26m-pose.pt` (with ByteTrack, `model.track()`) AND `yolo26m.pt` (filtered to `classes=[baseball_bat_id]`) on each video in the same loop. Export `keypoints_raw.csv` and `bat_boxes_raw.csv`.
- **Step 5:** Select the batter track by **minimum average wrist-to-bat distance** across frames (not size/persistence) — directly uses the bat detection output. Falls back to persistence + bbox size only if no bat is detected at all in a clip.

**Impact on roadmap:** Phase 2 (bat tracking) is no longer "build a bat detector from scratch" — it's now "evaluate pretrained bat detection quality, especially during fast-swing motion blur, and fine-tune `yolo26m.pt` on CVAT-annotated frames only if it fails systematically." Bat trajectory data (`bat_boxes_raw.csv`) is now available from Phase 1 onward, ahead of when Phase 5 metrics need it.

Add to pipeline package structure: `pipeline/bat/detector.py` (wraps pretrained `yolo26m.pt` bat-class inference) and `pipeline/pose/batter_selector.py` updated to take `bat_df` as an input.

---

### 2026-06-12 — Batter Selection Logic Revised: Largest-Bbox + Segment ID (supersedes bat-proximity)

**Decision:** Replaced bat-proximity-based selection with a simpler, validated rule, based on analysis of `keypoints_raw.csv` for `Chae_friend_Righty_Batting_V2.MOV` (444 frames, 18 track_ids, up to 3 people per frame):

- **Selection rule:** per frame, the batter is the track with the largest bounding-box area. Validated 100% accurate across all 444 frames.
- **Confidence > 0.9 rejected as a signal:** batter tracks averaged ~0.82-0.84 overall confidence (~0.91-0.93 on core joints), while a non-batter background track hit 0.94/0.97 — higher than any batter. Confidence is kept in the output as a per-keypoint quality field for smoothing, not for selection.
- **Multi-clip discovery confirmed:** `Chae_friend_Righty_Batting_V2.MOV` is 6 concatenated swings (track_ids 1, 9, 19, 34, 37, 43), each its own contiguous frame range. The selector now assigns a `segment_id` that increments whenever the selected track_id changes — this splits multi-swing video files into separate batting events automatically.
- **Bat detection decoupled from selection:** `bat_boxes_raw.csv` is no longer used to pick the batter. It remains useful downstream (e.g., bat-speed/association metrics in Phase 5) but Step 5 now only needs `keypoints_raw.csv`, so it can be re-run cheaply without re-running GPU inference.
- **Non-batter tracks are filtered out** of `keypoints_batter.csv` entirely.

**Implementation:** `pipeline/pose/batter_selector.py` created (`select_batter()`, `summarize_segments()`). Tested against `keypoints_raw.csv` — correctly split into 8 segments across 3 videos (V1s = 1 segment each, V2 = 6 segments), 17,952 raw rows -> 13,447 batter rows.

**Updated stage contract:** `pose.batter_selector`: input `keypoints_raw.csv` -> output `keypoints_batter.csv` (adds `segment_id`; drops `bat_boxes_raw.csv` as an input).

---

### 2026-06-12 — End of Day Summary & Next Steps

**Status today:**
- **Phase 1 (pose estimation validation): complete.** All key body keypoints track correctly across slow-motion and regular-speed clips. Multi-person batter selection validated (largest bbox per frame = batter, 100% accurate on test video).
- **Bat detection (part of Phase 1, Step 4): wired up.** `yolo26m.pt` filtered to the "baseball bat" class runs alongside pose+ByteTrack in the same pass, producing `bat_boxes_raw.csv`.
- **Phase 2 remaining work (reframed):** bat detection itself is done — what's left is evaluating pretrained bat-detection quality, especially during fast-swing motion blur, and deciding whether `yolo26m.pt` needs fine-tuning on CVAT-annotated frames.

**Identified gap — bat angle / 3D path is new scope, not yet in the pipeline:**
- Current bat detection = 2D bounding box only. No orientation/angle, no depth (z).
- To get a bat path with x/y/z and angle-to-ground, the pipeline needs: (1) bat **endpoint detection** (knob + tip, not just a box) for 2D orientation, and (2) a 3D lifting step for those points (MotionBERT lifts body joints only, not the bat).
- "Bat path at contact" also depends on **event detection (Phase 4)**, which itself depends on Phase 3's 3D output — so this can't be scoped into Phase 2.

**Revised near-term order:**
1. Finish Phase 2 — QA pretrained bat-bbox detection quality (motion blur on fast swings); decide fine-tune vs. not.
2. Phase 3 — body 2D->3D lifting (MotionBERT) on `keypoints_batter.csv` segments; separately scope bat-endpoint detection + 3D lifting as new work if bat angle/path metrics are wanted.
3. Phase 4 — event detection (toe touch, heel plant, launch, **contact**, extension) using 3D body (+ bat, once available).

**Next session starts at:** Phase 2 QA of bat detection quality across existing test clips.

---

### 2026-06-24 — Repo built from notebook + Stage 5 "Apply 2D Domain" shipped

**Decision:** The notebook `YOLO_Baseball.ipynb` is now a real repo. `src/` holds one module per
validated stage (pose, batter, motionbert setup/model/inference, save_3d, viz3d, viz2d) driven by
`config.py` + `run_pipeline.py`. Pose model aligned to `yolo26m`. Bat-detection QA and the
Kalman/wrist-proxy experiment are deliberately **excluded** (bat tracking stays deferred). Drive
mounts locally at `G:/My Drive` = Colab `/content/drive/My Drive`. Torch is CPU-only here and
flaky to import, so `run_pipeline` lazy-imports pose/motionbert — the analysis stages (batter,
viz2d, apply2d) run torch-free.

**New Stage 5 — `src/apply_2d_domain.py` (+ `colab/apply_2d_domain_colab.py`):** the agent team's
answer to "apply the 2D domain effectively." Instead of a plain skeleton it draws four swing
angles on the video and reports them live: hip-shoulder separation, spine tilt, front-knee angle,
and **attack-angle proxy** (smoothed hand-path direction at the peak-hand-speed "contact" frame).
Outputs `skeleton_2d_biomech_*.mp4` (8 clips) + `2d_metrics.json` (26 segments) to Drive. Agent
artifacts in `feedback-engine/` (MEDA→APA→APE) and copied to Drive.

**Key finding (Chae):** the lower half is already good (front-knee braces to 148–164° at contact);
the limiting variable is **repeatable hip-shoulder separation** — swings with sep ≥16° land attack
angle in the ideal 5–20° band (+10 to +11°), sep ≤10° goes flat/negative. So separation is the
lever that carries attack angle. **Caveat:** all values are 2D view-dependent proxies, and the
peak-speed "contact" heuristic misfires on some segments — proper **event detection (Phase 4)**
is the next dependency. Bat tracking (Phase 2.5, FMO/SAM2/TrackNet) + exact attack angle/bat speed
remain deferred.

---

### 2026-06-24 (later) — Phase 3 closed out (3D metrics) + bat tracking resumed (TrackNet)

**3D metrics shipped — `src/compute_3d_metrics.py` (+ `--stage metrics3d`, + Colab cells).**
Reads `keypoints_3d.json`, writes `metrics_3d.json` (8 Chae segments). These are the
view-independent versions of the swing angles. **Verified MotionBERT axis convention on our
output: vertical = z (index 2), UP = −z, horizontal plane = (x, y).** Hip-shoulder separation
must use the **undirected** line angle (min(θ, 180−θ)) — the directed angle reads ~147° for what
is really ~33°. Validated numbers (righty): separation peaks **55–61°**, lead-arm extension peaks
**120–141°** — both biomechanically sane. Attack-angle-3D still swings wildly because the
peak-hand-speed "contact" frame ≠ true contact → confirms **event detection (Phase 4)** is the
gating dependency for any at-contact metric (2D or 3D).

**Bat tracking — decision: TrackNet (user choice 2026-06-24).** Built
`colab/tracknet_bat_tracking_colab.py`: a TrackNetV2-style temporal U-Net (3 stacked frames →
knob+tip sigmoid heatmaps; forward-pass shape-verified), motion-blur augmentation as the key to
the contact zone, YOLO-box pseudo-labels + an in-Colab/CVAT manual labeler, WBCE training loop,
and inference → `bat_track.csv` + annotated MP4. **Not zero-shot — needs ~200–500 labeled frames
incl. contact-zone frames.** `colab/bat_attack_angle_colab.py` then gives the *exact* attack angle
from the tracked sweet spot (the 2D overlay's wrist-path value was the proxy for this).

**Repo:** Colab work now lives in `colab/` (estimation, 3D metrics, 2D overlay, bat tracking,
attack angle), mirrored to `Drive/colab/`. Still deferred: bat **speed**/swing length (need scale
calibration), Phase 4 event detection (now the top priority), React Native app.

**3D render fix + output convention (user request 2026-06-24):**
- **Convention:** all code that outputs a VIDEO writes to the **`Batting Diagnoses`** folder
  (`config.DIAGNOSES_DIR`; Colab `DIAG`). CSV/JSON stay in the project root / `Batting Key Point`.
  Applied to viz2d, viz3d, apply2d and all Colab cells.
- **3D simulation was rendering sideways/squished** because the old render used dim1 as "up." Fixed:
  vertical = dim2 (up = −dim2), dim0 = lateral, dim1 = depth, with a **front-on camera**
  (`elev=8, azim=-85`) so it matches the camera angle of the source video. `src/visualize_3d.py`
  now has `run_visualize_3d_from_json()` to re-render straight from `keypoints_3d.json` (no GPU).
  Re-rendered all of the user's own clips (Chae ×8 segs, outdoor_net, Indoor_Cage) into Batting
  Diagnoses; pro slow-mo 3D clips not yet re-rendered.

---

### 2026-08-11 — External prototype cloned: swinglens-prototype (on-device PWA reference)

**What:** Cloned `memekr/swinglens-prototype` into `src/swinglens-prototype/`, kept separate from
our own pipeline. It's a Next.js/React PWA that runs MediaPipe Pose Landmarker Lite entirely in
the browser, no server, no upload. Full summary and verified run instructions:
`2026-08-11-swinglens-prototype-summary.md`. Posted to the new `#baseball-project` Slack channel.

**Why it's here:** Reference for the on-device/PWA product direction and its competitor research
(10 products compared in `docs/competitive-analysis.md` inside the clone), not a replacement for
the YOLO26 → MotionBERT 3D pipeline in progress. Explicitly does not measure bat speed, exit
velocity, or true attack angle.

**Verified today:** `npm install`, `npm run build`, `npm run lint`, `npm test` (9/9), and
`npm run dev` (serves on localhost:3000) all pass cleanly. Repo was private and inaccessible to
the user's GitHub account until the owner made it public today.

---

### 2026-08-13 — swinglens-prototype: rebuilt on our body-labeling scheme + new phase model

**Body labeling:** `src/swinglens-prototype/lib/skeleton.ts` (new) converts MediaPipe's raw
33-point output into this project's canonical 17-point scheme from `Baseball Resources/body-labeling.md`
(head, torso, center hip, L/R shoulder/elbow/hand/hip-joint/knee/foot; torso and center hip are
derived midpoints). Every landmark-consuming module — geometry, analysis, drills, `SkeletonView`
rendering, the demo generator — now reads joints by name instead of raw MediaPipe indices. Hero
copy updated from "33 body landmarks" to "17."

**Phase model replaced:** setup/launch/contact/follow → **Trigger** (motion-onset detector: first
frame departing from the still start) → **Execution** (front-foot-plant detector: lead foot's most-
planted sample) → **Impact** (peak hand speed, same heuristic as the old "contact," now shows an
explicit "check whether the batter is squaring up" note tied to the video frame) → **Follow-through**
(unchanged timing, but now also computes a tracked hand-path polyline from Impact onward — the
honest proxy for bat barrel path since the bat isn't tracked — scored against a circular/slightly-
upward reference shape as a 5th metric, "Swing path shape"). Full definitions came from the user.

**Status:** All changes are local only, in `src/swinglens-prototype/` (not pushed — this account
has no write access to `memekr/swinglens-prototype`, and the live Vercel deploy is on memekr's
account). Verified via lint/test/build + manual browser check of the demo report. Not yet live at
swinglens-prototype.vercel.app.

---

### 2026-08-29 — RAG parent-child chunking

**Decision:** Ingest parses Markdown structurally and indexes 148 child chunks (~120–220 embedding-model tokens) with a Document/Section/Content-type prefix. Retrieval expands hits to a 300–600 token parent before the LLM. Hybrid eval on 22 IC questions: r@1 0.773, r@5 0.909, MRR 0.841 (IC-15 and IC-20 miss). RCTS 300/80 was r@1 0.955 on 416 slices; MRR dropped because mean-pooling dilutes needle sentences in larger children.

`rag_answer.py` generates with local Ollama `gemma4:e4b` (Anthropic helpers stay commented). It prints Sources/Retrieved-from after prose, and keeps citations out of the answer body. Do not rewrite coaching notes into simpler English; add alias lines for athlete phrasing (curveball, rolling over, athletic position) then rebuild ingest.

**Next:** alias edits on Top-Hand and Episode 5, rebuild, re-eval IC-15/IC-20. Generation is local Gemma 4 E4B via Ollama (`gemma4:e4b`); Anthropic is commented in `rag_answer.py`. Keep drill names exact. Golden `note` is still not an LLM prompt.

---

### 2026-08-27 — RAG corpus is RAG Resources

**Decision:** All `rag_*.py` scripts now read source notes from `Baseball Resources/RAG Resources/` and write Chroma/BM25 under `RAG Resources/rag_index/`. The old `transcripts_clean/chunks.jsonl` path is no longer the ingest source.

`rag_eval.py` only scores whether search found the right note. `rag_answer.py` is the live RAG app (retrieve, then local Gemma writes under `SYSTEM_PROMPT`). Abstention is a refuse-to-guess gate when cosine is below 0.42. Rerank is `Retriever.search(..., rerank=True)` in `rag_ingest.py` and is not wired into `answer_question`. Golden-file `note` text is never sent to the model. `ideal_answer` is a future grading key, not required to run answers.

Slack recaps for this workstation go to `#baseball-project` on the june-claude-ai workspace.

---

### 2026-08-31 — RAG generation is local Gemma 4 E4B

**Decision:** `rag_answer.py` writes answers with Ollama `gemma4:e4b` on localhost. Anthropic client code stays commented for a one-file revert. Hybrid retrieve (Chroma + BM25) is unchanged; only the generation step swapped.
