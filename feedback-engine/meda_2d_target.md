# MEDA — Mathematical & Engineering Development for Athlete
## Target: applying the 2D domain to batting video

**Agent:** MEDA (step 1 of MEDA → APA → APE)
**Run date:** 2026-06-24
**Question being answered:** *How do we effectively apply the 2D domain in the video?*
**Athlete profile (default):** High-school, right-handed, level / contact-style swing.
*(Profile fields — height, weight, stance, batting style — plug in per athlete; the
ranges below are the HS-level engineering targets.)*

---

## 1. What "applying the 2D domain" should mean

The raw 2D domain is the set of body points YOLO gives us in the flat image plane (17 joints
per frame, in pixels). A plain skeleton drawing uses that domain but extracts no information.

**Engineering target:** the 2D domain is applied *effectively* when each frame yields the
small set of **angles** that the kinetic chain of a swing is actually built from, drawn on the
video and reported as numbers a coach can read. We do this in 2D because four of the
highest-value swing quantities are *approximable from body keypoints alone* — the
"≈" tier in the Baseball Savant derivability matrix
(`2026-06-24-baseball-savant-batting-parameters.md`, Part 5). No bat tracking, no 3D required.

**Hard constraint (the honest part):** a 2D angle is a *projection*. Its value depends on the
camera view. So every number we produce is a **view-dependent proxy**, not a lab-grade 3D
reading. That is acceptable and expected at this layer — the exact versions come from
MotionBERT (3D) and, for true attack angle/bat speed, from bat tracking (Phase 2.5, deferred).

---

## 2. The target quantities, the physics, and the HS-level ranges

For each quantity: definition (plain language), why physics makes it matter, how it is computed
from 2D points, and the target band for a high-school hitter.

### 2.1 Hip–shoulder separation — unit: degrees
- **What:** how far the shoulders have turned *relative to* the hips. Think of winding a spring:
  the hips start to open while the shoulders stay closed.
- **Physics:** the torso muscles between hip and shoulder act like a stretched elastic band.
  Stretch (separation) before release stores elastic energy that is returned as rotational
  speed — the start of the kinetic chain that ends in bat speed. No stretch → arms-only swing.
- **2D computation:** angle between the shoulder line (L↔R shoulder) and the hip line (L↔R hip).
  Mapped to 0–90°. *(2D limitation: in a pure side view both lines foreshorten, so the reading
  understates true separation — read it as a trend, not an absolute.)*
- **HS target:** building toward **~25–45°** of separation at the launch point. Below ~20° = the
  hitter is "spinning out" (hips and shoulders turning together).

### 2.2 Spine tilt — unit: degrees from vertical
- **What:** how far the torso (mid-hip → mid-shoulder) leans away from straight up.
- **Physics:** the spine is the rotation axis. A *stable, repeatable* tilt keeps that axis
  consistent so rotation is efficient; tilt that drifts frame-to-frame leaks energy and moves
  the contact point around.
- **2D computation:** angle between the spine vector and image-vertical.
- **HS target:** not a "more is better" number — the target is **consistency** through the
  swing plus a slight tilt away from the plate at contact. Large frame-to-frame swings in this
  value are the red flag.

### 2.3 Front-knee angle — unit: degrees (180° = straight leg)
- **What:** the interior angle of the lead knee (front = left leg for a righty, right for a lefty).
- **Physics:** at contact a hitter wants to **brace** the front leg — stop its forward motion so
  the energy travelling up the body gets converted into rotation instead of drifting forward.
  A braced (straightening) front leg is a hallmark of efficient energy transfer; a soft,
  collapsing front knee bleeds power.
- **2D computation:** interior angle hip–knee–ankle on the front leg.
- **HS target:** deep flex during the load (energy stored, ~**100–140°**), then **firming toward
  ~150–170°** by contact. The *pattern* (load → brace) matters more than any single value.

### 2.4 Attack angle (wrist-path proxy) — unit: degrees (+ = swinging up)
- **What:** the direction the hands are travelling at the moment of contact. This is the
  body-keypoint proxy for MLB **attack angle** (officially the vertical angle of the bat's sweet
  spot at impact).
- **Physics & why it's the headline:** an incoming pitch arrives on a downward plane of about
  −5° to −20°. Matching that plane (swinging slightly up) maximizes the window of solid contact.
  MLB data shows a **62-point wOBA gap** between hitters in the ideal band and everyone else.
  It is also a **timing** metric — being early or late changes the angle at contact, so a bad
  number can mean bad timing, not just a bad swing shape.
- **2D computation:** smoothed velocity direction of the hands (average of both wrists), sign
  flipped so a rising path is positive. Reported at the **estimated contact frame** = the frame
  of peak hand speed (a lightweight stand-in for real event detection, Phase 4).
- **HS target:** **+5° to +20°** at contact (the official ideal band). < 5° = chopping down /
  early; > 20° = under the ball / late.

---

## 3. The reference table (what APA and APE measure against)

| Quantity | Unit | Poor | HS target |
|---|---|---|---|
| Hip-shoulder separation | ° | < 20 | ~25–45 at launch |
| Spine tilt | ° from vert. | drifts a lot | stable; slight tilt at contact |
| Front-knee angle | ° (180 = straight) | collapses (stays bent) | load ~100–140 → brace ~150–170 |
| Attack angle (wrist proxy) | ° (+ = up) | < 0 or > 25 | **+5 to +20 at contact** |

---

## 4. Engineering decision passed downstream

**Implement the 2D domain as an analysis overlay** (`src/apply_2d_domain.py`) that, per frame,
draws these four angles on the video and reports them in a live, color-coded HUD (green when in
the target band), flags the estimated contact frame, and writes a per-segment `2d_metrics.json`.
APA consumes that JSON to grade a real swing; APE turns the gap between this target and APA's
findings into drills.

→ Handoff to **APA** ([[apa_chae]]).
