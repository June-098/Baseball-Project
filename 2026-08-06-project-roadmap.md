# Baseball Mechanics AI — 12-Month Project Roadmap

**Version 1.0 · 6 August 2026 · Planning horizon: Aug 2026 – Jul 2027**

Team: 2 people, distributed (US / Korea). Combined capacity ~50 hrs/week (one full-time, one part-time).
Optimizing for: **a shippable product real athletes use.**

---

## 0. Executive Summary — The One Decision That Matters

The honest position is earlier than the repo history suggests. **One stage works end to end
today: the 2D biomechanics overlay.** Pose extraction runs but is only partially reliable.
Batter selection and swing segmentation are not working. 2D→3D lifting is not working. And
event detection — the thing that makes every at-contact number trustworthy — was never built.

That means the first quarter is not "add event detection to a working pipeline." It is
**restore a working pipeline, then add event detection.** Planning against the optimistic
version of this baseline is the fastest way to lose the year.

So the strategic call for the next 12 months is narrow and unglamorous:

> **Ship batting only, to real users, with numbers you can defend. Treat pitching as a
> validated technical spike, not a shipped product. Do not touch golf or tennis.**

The canvas vision — batting + pitching + tennis + golf, on iPhone + Android + web — is the
right 5-year vision and the wrong 12-month plan. Two part-time people across a 13-hour time
gap cannot ship four sports on three platforms. What they *can* do is ship one sport
extremely well and build the architecture so that sport #2 costs 30% of what sport #1 cost.
That compounding is the actual asset.

**The 12-month definition of success:**

| # | Outcome | Measured by |
|---|---|---|
| 1 | Batting analysis in real athletes' hands | 50+ athletes, 500+ analyzed swings, ≥40% return for a 2nd session |
| 2 | Numbers you can defend | Contact-frame timing MAE < 15 ms; key angles validated against OpenBiomechanics |
| 3 | Feedback that is cited, not hallucinated | 100% of coaching statements linked to a metric ID + video timestamp; 0 banned claims |
| 4 | Unit economics that survive scale | < $0.15 marginal cost per analyzed swing |
| 5 | Sport #2 is cheap | Pitching **spike** validated offline on shared infra; ≥70% code reuse demonstrated |
| 6 | **The pipeline cannot silently break again** | Stage-level regression tests green in CI; a deliberate break is caught automatically |

---

## 1. Goal

### Product goal

A single-camera phone app that turns a batting or pitching clip into body-mechanics data and
coaching feedback, priced so a high-school athlete can afford it without a facility, a radar
unit, or a wearable.

### Why this can win

Every incumbent requires hardware or a building. Hawk-Eye needs a stadium. HitTrax and Rapsodo
need purchased equipment. DiamondKinetics needs a bat sensor and still sees no body. The
un-served market is the athlete with a phone and a parent who is not spending $2,000. That is
the entire youth and high-school market, and nobody is serving it credibly.

### The non-negotiable principle

From your own vision document, and it should be framed on the wall:

> **Deterministic CV and geometry produce the measurements. The language model reviews,
> explains, and coaches. The language model never produces a number.**

The moment an LLM is allowed to estimate bat speed from pixels, the product becomes
unfalsifiable, un-reproducible, and — with minors — indefensible. This boundary is the
product's integrity and it is not a technical preference, it is policy.

---

## 2. Where You Actually Are (Honest Baseline, Aug 2026)

| Capability | State | Notes |
|---|---|---|
| **2D biomechanics overlay** | ✅ **Working** | **The one stage that works end to end.** 4 angles drawn live; `2d_metrics.json`. This is the product's current floor. |
| Local batch pipeline (183 clips) | ✅ Working | Streaming + subprocess batching, memory-bounded |
| Pose extraction (YOLO26m-pose + ByteTrack) | 🟡 **Partial** | Runs, but not reliable enough to build on. Needs a measured failure profile before anything downstream can be trusted. |
| Training corpus | 🟡 Partial | 173 Penn Action swings + 10 pro slow-motion clips |
| Feedback engine (MEDA/APA/APE) | 🟡 Conceptual | Roles defined; not running code |
| **Batter selection + swing segmentation** | ❌ **Not working** | Previously recorded as validated; it is not. Without it there is no reliable per-swing unit of analysis. |
| **2D→3D lifting (MotionBERT)** | ❌ **Not working** | Every view-independent metric is unavailable until this is restored. |
| **Event detection** | ❌ **Never built** | The gating dependency. Blocks every at-contact metric. |
| Bat tracking | ⏸ Deferred / failing | Pretrained YOLO collapses to 17–18% in the contact zone |
| Capture quality gate | ❌ Missing | No protection against unusable uploads |
| App | ❌ Not started | — |
| Pitching | ❌ Not started | Taxonomy and phase model documented in canvas only |

**Reading of this table:** roughly **25–30%** through the *measurement* problem and 0% through
the *product* problem. Several stages previously recorded as complete are not. **Q1 is a
recovery quarter, not an extension quarter.**

> ⚠️ **The documentation drift is itself a finding.** `MEMORY.md` records batter selection as
> "100% accurate across 444 frames" and 3D lifting as producing biomechanically sane angles
> (separation 55–61°). Neither holds today. Whatever the cause — regression, environment
> change, or a validation narrower than it appeared — **stage-level regression tests become a
> Q1 deliverable (M1.0)**. A pipeline that can silently go from "validated" to "not working"
> without anyone noticing will do it again, and next time it may happen after you have users.

**Check this before scoping anything:** the 2026-06-12 decision record explicitly *decoupled*
batter selection from bat detection — selection became largest-bbox-per-frame and
`bat_boxes_raw.csv` was dropped as an input. If segmentation is currently blocked by failing bat
tracking, that coupling may have been reintroduced by accident. If so, **decoupling it again is
potentially a same-day unblock rather than a quarter of work.** Worth an hour of investigation
before committing the quarter.

---

## 3. Physics Limits — Design Around These, Do Not Fight Them

These come from your own vision analysis and they cap what is honestly claimable. Every
milestone below respects them.

| Constraint | Consequence | Product response |
|---|---|---|
| 30 fps = 33.3 ms/frame; a 70–90 mph bat travels **1.0–1.34 m between frames** | Contact instant and peak bat speed are **not recoverable** at 30 fps | Capture gate **rejects or downgrades** 30 fps uploads for contact metrics |
| 120 fps → 0.26–0.34 m/frame; 240 fps → 0.13–0.17 m/frame | Even 240 fps needs sub-frame fitting | 120 fps = minimum for contact metrics; 240 fps recommended |
| Monocular 3D has ambiguous absolute scale and depth | "Metric" 3D from one phone is a claim you cannot support | Report angles (scale-free) confidently; gate distances/speeds behind calibration |
| 2D wrist-path angle ≠ 3D bat-barrel attack angle | Current "attack angle" is a proxy, not the MLB metric | Rename in UI to **"hand-path angle (proxy)"** until bat tracking lands |
| Exit velocity requires ball trajectory + time + scale | Cannot be derived from the bat or body alone | Do not ship exit velocity in Year 1. Say so plainly. |

**Management note:** the single highest-leverage non-code action in Q1 is writing a capture
protocol that gets users to shoot at 120 fps+ on a tripod. A better input beats a better model,
costs nothing, and no competitor can copy it away from you.

---

## 4. Technical Requirements — Batting

### 4.1 Athlete segmentation (drives the baseline comparison)

Per the canvas, three hitter archetypes, each with its own reference distribution:

| Archetype | Profile | Reference athletes |
|---|---|---|
| **Power** | High exit velocity, higher strikeout rate | Judge, Ohtani, Raleigh, Schwarber, Alvarez, Caminero, Wood |
| **Contact** | High contact rate, low strikeout rate | Lee Jung-hoo, Ichiro, Arraez, Hoerner, Simpson, Clement |
| **Gap / mid-power** | Doubles, balanced profile | Olson, Rafaela, Burleson, Devers, Freeman, Soto |

Comparing a contact hitter against a power-hitter baseline produces confidently wrong advice.
Archetype selection is a required user input, not optional.

### 4.2 Required user inputs

Archetype · height & weight · handedness (L/R) · batting video

### 4.3 Pipeline requirements

| # | Layer | Requirement | Status |
|---|---|---|---|
| 1 | **Capture gate** | Score resolution, true FPS, shutter blur, camera motion, full-body visibility, view angle, occlusion, lighting. Reject early, before expensive inference. | Build Q1 |
| 2 | **Pose** | 🟡 **Partially working — stabilize first.** Measure the failure profile (which joints, which views, which frame rates) before extending. COCO-17 is insufficient for hands/pelvis/thorax; extend only after it is reliable. | **Fix Q1** / extend Q2 |
| 3 | **Batter selection + segmentation** | ❌ **Not working — restore first.** Largest-bbox-per-frame selection, `segment_id` on track change. Verify it is not incorrectly coupled to failing bat detection. | **Fix Q1 — first task** |
| 4 | **Player isolation** | Instance segmentation mask to remove background people; batter identity score from uniform + box ROI + temporal keypoint quality, not ByteTrack ID alone. | Q1 |
| 5 | **Keypoint quality** | Per-keypoint confidence/covariance; explicitly tag `observed` / `interpolated` / `proxy`. Never blend them silently. | Q1 |
| 6 | **3D lifting** | ❌ **Not working — restore.** MotionBERT behind a swappable adapter. Separate camera-space / world-space / image-space by type. Mirror transform for L/R in exactly one place. **If it cannot be restored in 3 weeks, ship 2D-only and defer 3D to Q3** (see fallback below). | **Fix Q1** |
| 7 | **Event detection** | 8-state model: address → load → stride → foot plant → launch → **contact** → extension → finish. Inputs: body keypoints, bat axis, ball trajectory, audio impact transient. Every event carries a timestamp + confidence interval. | **Q1–Q2 — top new build** |
| 7 | **Metrics** | Hip-shoulder separation, spine tilt, front-knee angle, lead-arm extension, hand-path angle (proxy), time to contact. Each carries value, unit, space, event, confidence, CI, source joints/frames, definition version. | Q2 |
| 8 | **Bat tracking** | TrackNet-style temporal model on knob + tip, motion-blur augmentation, ~200–500 labeled frames **including contact-zone frames**. | Q3 (conditional) |
| 9 | **Blocked metrics** | Bat speed (needs scale calibration). Exit velocity (needs ball tracking). Do not ship in Year 1. | Year 2 |

### 4.4 Audio is underrated

Bat-ball contact produces a sharp audio transient that a phone mic captures at 44.1 kHz —
roughly **1,470× the temporal resolution of a 30 fps video frame**. Audio will not tell you
*where* contact happened, but it pins *when* to within a millisecond. For a project whose
central weakness is contact timing, this is the cheapest accuracy win available. Prototype it
in Q1 alongside the visual event model.

---

## 5. Technical Requirements — Pitching

Pitching reuses the skeleton backbone but is **not** a configuration change. Three things are
genuinely different: the event model, the arm-slot taxonomy, and the risk profile.

### 5.1 Arm-slot segmentation

| Slot | Description | Reference pitchers |
|---|---|---|
| **Overhand** | Near-vertical delivery, downward plane | Schlittler, Cole, Yesavage, Chapman, Miller |
| **Three-quarter** | Between overhand and sidearm; most common | Ohtani, Woo, Castillo, Yamamoto |
| **Sidearm** | Near-horizontal arm path | — |
| **Underhand / submarine** | Low-to-high, rising path | — |

Arm slot should be **auto-classified from the video and confirmed by the user**, not typed in
blind — most amateur pitchers misreport their own slot.

### 5.2 Kinetic chain (the thing being measured)

Ground → legs → pelvis → trunk → shoulder → elbow → hand → release.

Pitching analysis is fundamentally about **sequencing and timing**, not static positions. The
question is never "is the elbow in the right place" but "did the pelvis start rotating before
the trunk, and by how many milliseconds." This makes event timing even more central than in
batting.

### 5.3 The 9 events

first movement · peak leg lift · hand separation · max stride / foot hover · **lead-foot
contact** · **maximum external rotation (MER)** · **ball release** · max lead-knee extension ·
finish

Two of these are hard from a single phone camera and must be flagged as such:

- **MER** — peak shoulder external rotation happens in a few milliseconds with severe forearm
  motion blur. Below 240 fps, treat as unmeasurable.
- **Ball release** — needs either ball detection or a very high frame rate. Without it,
  release-point consistency (arguably the single most useful pitching metric) is not available.

### 5.4 Multi-pitch requirement — a product constraint, not a technical one

Your own canvas states it: judge across several pitches, never one delivery. A single-pitch
pitching report is professionally misleading. **The pitching product must require a minimum of
5 pitches per session** and report variance, not just the mean. Release-point *consistency* is
the deliverable; release point is only an input.

### 5.5 Safety policy — decide this before writing pitching code

You will be analyzing throwing mechanics for minors. Pitching mechanics are causally linked in
public discourse to elbow and shoulder injury. Therefore:

> **The product must never output injury risk, injury prediction, or medical language — not
> as a hedge, not as a percentage, not as a "flag." No ground truth exists in your data to
> support such a claim, and the population is children.**

Ship descriptive mechanics ("your trunk begins rotating 40 ms before lead-foot contact,
earlier than your archetype baseline") and never diagnostic claims. Encode this as a banned-
phrase list enforced in the output schema validator, not as a guideline anyone has to remember.

### 5.6 Shared vs. sport-specific — the reuse thesis

| Reusable across sports (~70%) | Sport-specific (~30%) |
|---|---|
| Capture quality gate | Event state model |
| Pose extraction + tracking | Metric definitions |
| Player isolation + identity | Archetype taxonomy |
| 3D lifting adapter | Baseline reference library |
| Evidence Packet schema | Coaching rubric |
| Feedback engine (MEDA/APA/APE) | Capture protocol specifics |
| App shell, auth, storage, billing | — |

**This table is the business case for the multi-sport vision.** Golf and tennis are, structurally,
"new event model + new metric definitions + new baseline library" on top of an existing spine.
Prove that with pitching in Q4 and the expansion story becomes credible to a partner, an
investor, or a hire.

---

## 6. Technical Suggestion — The AI Agent / Multimodal Architecture

### 6.1 The architectural boundary

```
┌─────────────────────────────────────────────────────────────┐
│  DETERMINISTIC LAYER  — produces every number               │
│  capture gate → pose → tracking → events → 3D → metrics     │
│  Output: Evidence Packet (JSON, schema-validated)           │
└───────────────────────────┬─────────────────────────────────┘
                            │  numbers cross this line ONE WAY
┌───────────────────────────▼─────────────────────────────────┐
│  REASONING LAYER  — explains, never measures                │
│  MEDA → APA → APE, backed by local VLM lanes                │
│  Output: coaching report, every claim citing an evidence_id │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 The Evidence Packet — the single most important interface

Everything the reasoning layer sees, it sees through this contract. Nothing else.

```json
{
  "run_id": "2026-08-06-athlete042-swing03",
  "athlete_profile": { "bats": "R", "age_band": "HS", "archetype": "contact",
                       "height_cm": 178, "weight_kg": 74 },
  "capture_quality": { "fps": 240, "view": "side", "score": 0.91,
                       "flags": ["contact_blur"] },
  "events": [ { "name": "foot_plant", "t_ms": 610, "confidence": 0.88 },
              { "name": "contact",    "t_ms": 842, "confidence": 0.76 } ],
  "metrics": [ { "id": "m_hss_01", "name": "hip_shoulder_separation",
                 "value": 31.2, "unit": "deg", "space": "world",
                 "event": "launch", "ci95": [27.8, 34.9],
                 "confidence": 0.83, "definition_version": "1.2",
                 "source_joints": ["L_hip","R_hip","L_shoulder","R_shoulder"] } ],
  "baseline_comparison": [ { "metric_id": "m_hss_01", "cohort": "contact_HS",
                             "cohort_p50": 38.4, "cohort_iqr": [32.1, 44.0],
                             "percentile": 24 } ],
  "keyframes": ["address","foot_plant","launch","contact","finish"],
  "overlays": ["pose","hand_path","uncertainty"]
}
```

**Design rules:**

1. If a metric is not in the packet, the agent cannot discuss it. This alone eliminates most
   hallucination.
2. Every metric carries a CI. The agent must soften or abstain when CI is wide.
3. `baseline_comparison` is computed deterministically by retrieval — **not** by the LLM.
4. Schema-validate on the way in and on the way out. Retry on failure, abstain after 2 retries.

### 6.3 Two-lane local VLM

Per your hardware analysis (M5 Max, 40-core GPU, 128 GB unified memory):

| Lane | Model | Size | Role |
|---|---|---|---|
| **Fast** | `google/gemma-4-12B-it-qat-q4_0-gguf` | ~6.98 GB | Capture-quality QA, keyframe description, draft reports, UI responses |
| **Strong** | `mlx-community/Qwen3.6-35B-A3B-4bit` | ~20.4 GB | Escalation only: low confidence, L/R confusion, multi-swing comparison, final coaching synthesis |

Both fit in memory; do not co-resident them at first. Keep the fast lane resident, load the
strong lane on demand. **Escalation should be < 20% of runs** — if it exceeds that, the
deterministic layer is under-performing and that is where the fix belongs, not in a bigger model.

Do not accept vendor benchmarks. Confirm lane assignment on your own golden set, measuring:
Korean and English coaching accuracy, frame ordering, left/right mirroring, temporal causality,
JSON schema compliance, and hallucination rate.

### 6.4 The three agents as running code

Today MEDA/APA/APE are conceptual roles. Make them services with typed contracts:

| Agent | Input | Output | Implementation note |
|---|---|---|---|
| **MEDA** — target definition | Athlete profile (archetype, height, weight, handedness, age band) | Target mechanics envelope + rationale | **Mostly retrieval, not generation.** Query the versioned baseline library for the matching cohort. The LLM writes prose around retrieved numbers; it must never invent the numbers. |
| **APA** — current-state analysis | Evidence Packet | Pros/cons by mechanical category, each citing metric IDs | Deterministic comparison first, LLM narrates second. |
| **APE** — recommendations | MEDA target + APA analysis | Ranked, athlete-specific corrections + drills | Must state *why* each correction follows from this athlete's specific gap. Generic advice = a failed run. |

**Guardrails, enforced in code:**

- Output schema requires an `evidence_id` on every claim; unparseable output is rejected, not patched.
- Banned-claim list (injury risk, medical language, "ideal," certainty language) checked post-generation.
- Abstain path: if capture score or metric confidence is below threshold, return "re-shoot"
  guidance instead of analysis. **An honest abstention is a feature.** It is also the cheapest
  possible response.

### 6.5 Baseline Reference Library — the piece that makes feedback credible

The canvas promises feedback derived from professional samples. Implement that as a
**versioned metric distribution store**, not video similarity search:

```
baseline_library/
  v1.2/
    batting/
      power_HS.json        # p10/p25/p50/p75/p90 per metric, n, source, capture conditions
      contact_HS.json
      gap_HS.json
      power_pro.json       # aspirational reference tier
    pitching/
      overhand_HS.json
      three_quarter_HS.json
```

Comparison is then a percentile lookup — fast, explainable, testable, and versioned so a
baseline change never silently rewrites past reports. Store cohort `n` and surface it: a
cohort of 6 athletes must not be presented with the authority of a cohort of 300.

### 6.6 Cost architecture

Local-first inference on the M5 Max is the right call for development and for the pro tier.
For consumer scale, the economics that matter:

- Deterministic CV is the expensive part; VLM reasoning over a small JSON packet is cheap.
- Cache aggressively — the same athlete re-analyzed should not re-run pose extraction.
- The capture gate is a **cost control**, not just a quality control: rejecting a bad upload in
  200 ms instead of running a full pipeline is the single largest lever on unit economics.
- Target < $0.15 per analyzed swing. Instrument this from Q2, not Q4.

---

## 7. Video Collection Strategy — Building Local Model Baselines

You need three distinct things, and conflating them is a common and expensive mistake:

| Need | Purpose | Volume | Quality bar |
|---|---|---|---|
| **A. Training data** | Fine-tune pose/event/bat models | Thousands of frames | Varied, realistic, messy |
| **B. Ground truth** | Validate metric accuracy | Hundreds of trials | Lab-grade, instrumented |
| **C. Baseline library** | Cohort comparison distributions | 30+ per cohort | Consistent capture protocol |

### Tier 1 — Public datasets (free, start here)

| Source | Contents | Use |
|---|---|---|
| **Driveline OpenBiomechanics** | **100 pitchers + 98 hitters**, C3D marker data: 47 body markers (22 lower-limb/pelvis, 25 head/upper-limb/trunk) **plus 10 bat markers**. Free for educational use through 31 Dec 2030. | **This is your ground truth (Need B).** The single most valuable external asset available to you. Validate hip-shoulder separation, sequencing, and lead-arm extension against real marker-based mocap. |
| **Penn Action** | 2,326 clips / 15 actions; **173 baseball_swing sequences already extracted** to `Penn_Action/batting/`, 7,561 frames, 13 joints + bbox + visibility | Pose/event pre-training and pipeline regression (Need A). Note: low resolution, ~30 fps, uncontrolled — good for robustness, useless for contact metrics. |
| **MLB-YouTube** (`piergiaj/mlb-youtube`) | Broadcast segments, activity-labeled | Raw video source only; needs pseudo-labeling + manual correction |
| **SportsPose / AthleticsPose** | 3D sports pose datasets | Cross-domain 3D lifting validation |

**Priority action:** license-check and download OpenBiomechanics in Week 1. Everything in your
validation gate depends on having real ground truth, and you currently have none.

### Tier 2 — Your own capture (highest value per unit effort)

Public data will never match your capture conditions. 30–50 athletes shot to your own protocol
beats 5,000 scraped broadcast clips for baseline purposes.

**Capture protocol (write this in Q1, it is a deliverable):**

- Fixed tripod, 1080p minimum, **120 fps minimum / 240 fps preferred**
- Two views where possible: side (sagittal) and front (frontal)
- A known-length reference object in frame (a bat of known length works) for scale
- Preserve original timestamps; when converting variable to constant frame rate, keep a
  **time mapping table**, never bare frame indices
- 5+ swings or pitches per session
- Consistent lighting, clean background, full body visible throughout

**Sourcing, in ascending order of effort:**

1. Yourselves and friends — start this week, zero coordination cost
2. Local high-school teams and academies — offer free analysis in exchange for footage rights
3. Korea-side advantage: your partner can run a parallel cohort in a different baseball
   culture. **Do not treat this as redundancy — treat it as your generalization test.** A model
   that works in both the US and Korea is meaningfully more robust than one tuned to a single
   region, and this is a real technical benefit of a distributed team.
4. Youth showcases and tournaments — high volume, but consent overhead is real

**Consent and minors:** get written parental consent before any footage of a minor is captured,
stored, or used for training. Build the consent form in Q1 alongside the capture protocol, not
after you have 200 clips you cannot legally use.

### Tier 3 — Partnership (Q3+)

A single relationship with a facility that runs Rapsodo, HitTrax, or Blast Motion gives you
paired video + instrument readings — exactly the ground truth needed to validate bat speed and
exit velocity later. Trade free analysis for data rights. One good facility partnership is
worth more than a year of scraping.

### Labeling

- **Tool:** CVAT (already in your stack)
- **Event labels:** 200–500 swings labeled with the 8 batting events; this is the training set
  for Q1's event detector and it is on the critical path
- **Bat knob/tip:** 200–500 frames **including contact-zone blur** — bootstrap from YOLO boxes,
  hand-correct the blurred frames
- **Practical note:** event labeling is ~2–4 min per swing. 300 swings ≈ 15–20 hours. This is
  well-suited to the part-time partner, is fully async, and needs no shared working hours.

### Legal caution

Broadcast footage (MLB, YouTube) is fine for internal research and model development.
Redistributing it, or showing it inside a commercial product as reference material, is not.
Keep two clearly separated buckets — `research_only/` and `commercial_ok/` — from day one.
Retrofitting that separation later is painful.

---

## 8. Major Milestones & Timeline

### Q1 · Months 1–3 (Aug–Oct 2026) — "Restore the Foundation"

**Theme:** ⚠️ **Recovery quarter.** Three broken stages must work again before anything new is
built on top of them. No product work. Event detection **starts** here but is expected to
finish in Q2 — that is the cost of the revised baseline, and pretending otherwise would just
move the failure later.

**Sequencing matters: do these in order. Each one gates the next.**

| Milestone | Deliverable | Owner |
|---|---|---|
| **M1.0 Regression harness** | Per-stage smoke tests + a 10-clip fixture set that fails loudly. **Build this first** — it is why the drift went unnoticed. | FT |
| **M1.1 Batter selection + segmentation restored** | Largest-bbox selection + `segment_id` working again. **First check whether it is wrongly coupled to failing bat detection** — that may be a same-day fix. | FT |
| **M1.2 Pose reliability profile** | Quantify where pose fails (joint, view, fps, occlusion). Stabilize to a known-good baseline. No extension work yet. | FT |
| **M1.3 3D lifting restored** | MotionBERT producing sane angles again, behind a swappable adapter. **3-week timebox — then take the fallback.** | FT |
| M1.4 Ground truth acquired | OpenBiomechanics downloaded, license-verified, loaded, mapped to your joint schema | PT |
| M1.5 Capture protocol + consent | Written protocol, example videos, parental consent form | PT |
| M1.6 Golden set labeled | 200–500 swings, 8 events each, in CVAT | PT |
| M1.7 Capture quality gate | Scores 8 dimensions, rejects before inference | FT |
| M1.8 Metric schema v1 | Every metric carries value/unit/space/event/CI/version | FT |
| M1.9 Event detection **started** | Audio transient prototype + visual state model v0. Completion target moves to Q2. | FT |

**Q1 exit gate — all five:**
1. Regression harness green, and it demonstrably catches a deliberately introduced break
2. Batter selection + segmentation working on the full 183-clip corpus
3. Pose failure profile documented with a known-good operating envelope
4. 3D lifting restored **or** the 2D-only fallback formally taken
5. Capture gate correctly rejects known-bad uploads

> **🔀 Fallback if 3D cannot be restored in 3 weeks:** ship the **2D-only product**. Your 2D
> overlay works today, and hip-shoulder separation, spine tilt, and front-knee angle are
> genuinely coachable from 2D alone with a stated view-dependence caveat. 3D moves to Q3 as an
> accuracy upgrade. **This is a perfectly good product** — do not let a broken 3D stage hold the
> whole year hostage. Most competitors at this price point offer nothing comparable.

---

### Q2 · Months 4–6 (Nov 2026–Jan 2027) — "Trustworthy Feedback"

**Theme:** finish event detection, then turn numbers into coaching. First real users.

| Milestone | Deliverable | Owner |
|---|---|---|
| **M2.0 Event detection shipped** | `events.json` with per-event timestamp + CI; **contact MAE < 15 ms at 240 fps**. Carried over from Q1. | FT |
| M2.1 Evidence Packet v1 | Schema, validator, generator wired to the pipeline | FT |
| M2.2 Baseline library v1 | 3 batting archetypes × HS tier, from own + public capture | PT |
| M2.3 Fast lane live | Gemma 4 12B running locally; capture QA + draft reports | FT |
| M2.4 MEDA/APA/APE as services | Typed I/O, schema validation, banned-claim enforcement, abstain path | FT |
| M2.5 Validation report | Your metrics vs. OpenBiomechanics; **unsupportable claims removed from all copy** | Both |
| M2.6 **Closed alpha** | **10 real athletes**, full loop, structured feedback collected | Both |
| M2.7 Cost instrumentation | Per-swing cost tracked from day one | FT |

**Q2 exit gate:** contact MAE < 15 ms; 10 athletes have received reports; ≥7 rated the feedback
"specific and useful"; 100% of claims carry an evidence citation; 0 banned claims emitted.

---

### Q3 · Months 7–9 (Feb–Apr 2027) — "Make It a Product"

**Theme:** app, scale, retention.

| Milestone | Deliverable | Owner |
|---|---|---|
| M3.1 **Platform decision** | **Pick ONE.** Recommendation: mobile web (PWA) first — one codebase, no app-store review, instant iteration. Native later, driven by evidence. | Both |
| M3.2 App v1 | Upload → capture-gate feedback → results → overlay playback synced to metrics | FT |
| M3.3 Backend + storage | FastAPI, S3, auth, consent capture, retention policy | FT |
| M3.4 Progress tracking | Session-over-session comparison — **the #1 retention driver** | PT |
| M3.5 Baseline library v2 | 30+ athletes per cohort; pro tier added | PT |
| M3.6 3D lifting (if deferred from Q1) | Restore as an accuracy upgrade on a working 2D product | FT |
| M3.7 Bat tracking | ❌ **Cut from the 12-month plan.** With the revised baseline there is no capacity for it. Year 2. | — |
| M3.8 **Open beta** | **50+ athletes**, self-serve | Both |

**Q3 exit gate:** 50 athletes onboarded; ≥40% return for a second session; cost/swing < $0.15;
p95 turnaround < 5 min.

---

### Q4 · Months 10–12 (May–Jul 2027) — "Harden Batting, Spike Pitching"

**Theme:** ⚠️ **Revised.** The pitching *alpha* is cut; a pitching **technical spike** replaces
it. With a recovery quarter absorbed at the start of the year, shipping a second sport would
mean shipping both badly. Batting retention is the more valuable asset at month 12.

| Milestone | Deliverable | Owner |
|---|---|---|
| M4.1 **Batting hardening** | Reliability, turnaround, cost, and retention work on the live product. **This is the priority.** | FT |
| M4.2 Pitching spike | 9-event model + arm-slot classifier running on the shared backbone, **validated offline against OpenBiomechanics pitching data. No users.** | FT |
| M4.3 Safety policy | Banned-claim validator built and tested before any pitching output is ever shown | Both |
| M4.4 Pitching baselines | Overhand + three-quarter, HS tier, seeded from OpenBiomechanics | PT |
| M4.5 **Reuse report** | Quantified code reuse from the spike; expansion cost model for pitching, golf, tennis | Both |
| M4.6 Year-2 plan | Written on Year-1 evidence, not on the original vision doc | Both |

**Q4 exit gate:** batting product stable with ≥40% retention holding at scale; pitching spike
validated offline; **≥70% infrastructure reuse demonstrated** (the extensibility claim proven
without the cost of a second launch).

> **Why cut the pitching alpha rather than compress batting?** A batting product 50 athletes
> use weekly is a real asset — for revenue, for recruiting, and for any funding conversation.
> A pitching alpha with 10 users and a shaky batting product is neither. The reuse report
> still proves the multi-sport thesis; it just proves it with evidence instead of a launch.

---

### Timeline at a glance

```
        Q1 Aug-Oct         Q2 Nov-Jan         Q3 Feb-Apr         Q4 May-Jul
        RESTORE            TRUST              PRODUCT            HARDEN + SPIKE
FT      regression tests   event detection    app v1             batting hardening
        batter selection   evidence packet    backend            pitching spike
        pose stabilize     agent services     3D (if deferred)   (offline, no users)
        3D lifting fix     VLM fast lane      progress tracking  safety validator
        capture gate       validation report
PT      golden set label   baseline lib v1    baseline lib v2    pitching baselines
        capture protocol   alpha recruiting   beta recruiting    reuse analysis
USERS   0                  10 (closed)        50+ (open beta)    50+ batting only
GATE    pipeline restored  MAE<15ms, 7/10     40% return         70% reuse proven
```

**What changed from v1.0 and why:** the revised baseline absorbs a full recovery quarter.
Event detection slips Q1→Q2, the alpha slips Q2 (unchanged, but tighter), bat tracking is cut
outright, and the pitching alpha becomes an offline spike. **Nothing was compressed to preserve
the original dates** — that would only relocate the failure to a worse quarter.

---

## 9. Key Deliverables Summary

**Documents:** capture protocol · parental consent form · metric definitions (versioned) ·
validation report vs. OpenBiomechanics · banned-claim policy · Year-2 plan

**Data:** golden set (200–500 labeled swings) · baseline reference library v1/v2 ·
own-capture corpus (US + Korea) · OpenBiomechanics-derived validation set

**Code:** event detector · capture quality gate · Evidence Packet schema + validator ·
MEDA/APA/APE services · baseline comparison engine · app v1 · FastAPI backend ·
pitching event model + arm-slot classifier · (conditional) TrackNet bat tracker

**Product:** closed alpha (10) → open beta (50+) → pitching alpha (10)

---

## 10. Limitations & Constraints — Two People, Two Continents

### 10.1 The honest constraints

| Constraint | Real consequence |
|---|---|
| **~50 hrs/week combined** | ~1 major workstream + 1 support workstream. Not 4 sports, not 3 platforms. |
| **13–14 hour time gap (US ↔ Korea)** | Near-zero overlap hours. Anything requiring live debugging together will stall for a day per exchange. |
| **Single points of failure** | Two people means two bus factors of 1. Illness, a job change, or burnout stops the project outright. |
| **No specialist depth** | Neither of you is a full-time biomechanist, iOS engineer, or ML researcher. Some problems will take 3× longer than they would for a specialist. |
| **No ground-truth hardware** | No force plate, no marker mocap, no radar. You are dependent on public datasets and partnerships for validation. |
| **Labeling is a hard floor** | 300 labeled swings ≈ 15–20 hours of human work that cannot be automated away or bought cheaply at this scale. |

### 10.2 Turning the time gap into an advantage

The 13-hour gap is usually framed as pure cost. Structured correctly it is close to
continuous progress — but only under strict discipline:

- **Split by ownership, never by task.** The full-time person owns the pipeline; the part-time
  person owns data, labeling, baselines, and user recruiting. Interfaces between them are files
  and schemas, not conversations.
- **Contracts before code.** Agree the JSON schema first, then both sides build against it
  independently. This is why the Evidence Packet matters organizationally as much as technically.
- **Async by default.** Written decision records (you already have MEMORY.md — keep it current),
  recorded walkthroughs instead of live demos, PR descriptions that assume the reader is asleep.
- **One weekly sync, non-negotiable.** 60 minutes, agenda written in advance, decisions logged.
  One real meeting beats five ad-hoc ones that only one person can attend.
- **Korea cohort as generalization test.** Two capture populations in two baseball cultures is a
  genuine technical asset. Most two-person teams cannot claim geographic diversity in their
  training data.

### 10.3 What to explicitly NOT do in the next 12 months

Scope discipline is the highest-value management act available to you. Written down so it can
be pointed at when temptation arrives:

- ❌ Golf and tennis — architected for, not built
- ❌ Native iOS + Android + web simultaneously — one platform, chosen on evidence
- ❌ Exit velocity and bat speed — physics blocks them without calibration and ball tracking
- ❌ Injury-risk claims — no ground truth, and the users are minors
- ❌ Training pose models from scratch — fine-tune only on demonstrated systematic failure
- ❌ Real-time on-device inference — server-side until the pipeline is proven
- ❌ Multi-camera capture — kills the core "just a phone" value proposition

### 10.4 Growing the team — how to get help

**Phase 1 (Q1–Q2): Advisors, not employees.** Cheap, high leverage.

- **One biomechanics advisor** — a college or high-school strength coach, or a kinesiology
  grad student. Offer free analysis and co-authorship on any writeup. This is the highest-ROI
  relationship available: it validates your metric definitions and prevents you from confidently
  shipping something a domain expert would immediately reject.
- **2–3 coach design partners** — they tell you whether feedback is actually actionable, which
  no benchmark can.

**Phase 2 (Q2–Q3): Build in public.** Recruiting follows attention; attention follows artifacts.

- Publish the validation work — "here is our contact-detection error vs. marker-based mocap" is
  exactly the kind of post that reaches CV engineers who care.
- Open-source a genuinely useful, non-core piece: the capture quality gate, or Penn Action
  event-labeling tools. Signals competence, costs no moat.
- Present at a sports-analytics venue (SABR, SSAC). The baseball analytics community is unusually
  open and actively looks for this kind of work.
- **Korea advantage:** the Korean baseball market is large, underserved by English-language
  tools, and your partner is positioned there. A Korean-language product is real differentiation,
  not a translation task.

**Phase 3 (Q3–Q4): Targeted contributors.** Only after there is something worth joining.

| Gap | Who | Engagement |
|---|---|---|
| Labeling throughput | Sports-science students | Paid hourly or course credit |
| Mobile/frontend | Contract RN or PWA dev | Fixed-scope, 4–6 weeks |
| Biomechanics validation | Kinesiology grad student | Research collaboration |
| Data partnerships | Facility owner | Data-for-analysis trade |

**What makes people join a two-person project:** a working demo, a clearly scoped problem, and
evidence you ship. You will have the first by Q2 and the third by Q3. Recruit then — not now,
when the honest pitch is still "we have a pipeline and a plan."

**Funding note:** the Q2 alpha results and the OpenBiomechanics validation report are, together,
a credible pre-seed narrative if you want one. Whether to raise is a Q3 decision informed by
whether retention holds — not a Q1 distraction.

---

## 11. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R0 | **Broken stages take longer than Q1 to restore** | **High** | **Critical** | Timebox each fix (3 weeks for 3D). Take the 2D-only fallback rather than extending. The 2D overlay already works — it is a shippable floor. |
| R0b | **Further silent regressions appear** | Medium | High | M1.0 regression harness is the first deliverable of the year, specifically because this already happened once |
| R1 | Event detection misses the < 15 ms target | Medium | **Critical** — blocks at-contact metrics | Audio fusion as a parallel path; if both fail, ship only non-contact metrics (separation, spine tilt, knee angle remain valid) |
| R2 | Users shoot at 30 fps regardless of guidance | **High** | High | Capture gate + in-app camera with FPS enforcement; degrade gracefully to non-contact metrics rather than refusing service |
| R3 | Labeling becomes the bottleneck | High | Medium | Start Week 1; active learning to prioritize informative frames; budget for paid student labelers in Q3 |
| R4 | One person becomes unavailable | Medium | **Critical** | Written decision records, no undocumented tribal knowledge, both sides able to run the full pipeline |
| R5 | Metrics fail validation against OpenBiomechanics | Medium | High | Better to learn this in Q1 than post-launch. Narrow claims to what survives; that is a successful outcome, not a failure |
| R6 | Bat tracking consumes the year | — | — | ✅ **Resolved by cutting it.** Removed from the 12-month plan entirely. Body mechanics alone is a viable product. |
| R7 | Scope creep toward golf/tennis | **High** | High | Section 10.3 exists to be pointed at |
| R8 | Youth data/privacy misstep | Low | **Critical** | Consent forms from Q1; minimize retention; never train on non-consented footage |

---

## 12. The Four Things That Matter Most

If everything else is noise, these are the load-bearing decisions:

1. **Restore the pipeline before extending it — and make it impossible to break silently
   again.** Three stages that were recorded as working are not. Build the regression harness
   first, then fix batter selection, pose, and 3D lifting in that order. Event detection is
   still the gate on every at-contact metric, but it cannot be built on a foundation that
   does not hold.

2. **Never let the model produce a number.** The deterministic/reasoning boundary is what makes
   this product defensible. Protect it in code, not in convention.

3. **Ship one sport, on one platform, to real users.** A batting product 50 athletes use weekly
   is worth vastly more than four half-built sports — as a business, as a technical asset, and
   as a recruiting story.

4. **Say what you cannot measure.** Physics blocks exit velocity, bat speed, and true 3D attack
   angle from a single uncalibrated phone. Competitors will overclaim these. Being the tool that
   is honest about uncertainty is a durable differentiator with coaches — the people who
   ultimately decide whether athletes adopt you.

---

## Sources

- [The OpenBiomechanics Project](https://www.openbiomechanics.org/)
- [drivelineresearch/openbiomechanics — GitHub](https://github.com/drivelineresearch/openbiomechanics)
- [The OpenBiomechanics Project: Driveline Goes Open Source](https://www.drivelinebaseball.com/2022/12/openbiomechanics-project/)
- [Analysis of Baseball Hitters at Different Levels of Competition — HAS-Motion](https://has-motion.com/wiki/doku.php?id=sift:tutorials:openbiomechanics_project:analysis_of_baseball_hitters_at_different_levels_of_competition)
- [AthleticsPose: Authentic Sports Motion Dataset](https://arxiv.org/pdf/2507.12905)
- [Monocular 2D Baseball Swing Pose Estimation](https://sensors.myu-group.co.jp/sm_pdf/SM4388.pdf)
- Internal: `Baseball Big Picture.canvas`, `05-vision-local-llm-roadmap.md`, `CLAUDE.md`, `MEMORY.md`, `README.md`
