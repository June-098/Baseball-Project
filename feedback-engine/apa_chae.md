# APA — Athlete Performance Analyzer
## Subject: Chae (friend) — batting, 2D-domain read

**Agent:** APA (step 2 of MEDA → APA → APE)
**Run date:** 2026-06-24
**Input:** `keypoints_batter.csv` + the 2D metrics from `src/apply_2d_domain.py`.
**Measured against:** [[meda_2d_target]].
**Clips:** `Chae_friend_Righty_Batting_V1` (1 swing), `Chae_friend_Righty_Batting_V2`
(6-swing BP), `Chae_friend_Lefty_Batting_V1` (1 long clip). Right-handed is the primary side.

> All numbers are **2D view-dependent proxies** at the **estimated** contact frame (peak hand
> speed). Treat them as directional, not lab-exact.

---

## 1. The numbers (at estimated contact)

| Clip / swing | Attack angle | Hip-shoulder sep | Front-knee | Spine tilt |
|---|---|---|---|---|
| Righty V1 · seg0 | **+10.2°** ✅ | 18.2° | 161.6° | 19.7° |
| Righty V2 · seg0 | −8.1° ✕ | 10.3° | 148.0° | 15.9° |
| Righty V2 · seg1 | **+10.1°** ✅ | 21.8° | 163.7° | 18.3° |
| Righty V2 · seg2 | −169.5° ⚠️ | 5.2° | 174.6° | 3.6° |
| Righty V2 · seg3 | **+11.1°** ✅ | 16.3° | 162.4° | 14.7° |
| Righty V2 · seg4 | +3.2° ✕ | 6.1° | 151.5° | 14.4° |
| Righty V2 · seg5 | −2.4° ✕ | 10.8° | 148.5° | 15.7° |
| Lefty V1 · seg0 | −47.4° ⚠️ | 11.4° | 154.6° | 4.6° |

✅ in the 5–20° ideal band · ✕ outside the band · ⚠️ unreliable contact estimate (see §4)

Across-swing ranges (righty): hip-shoulder sep cycles **0–~85°**, front-knee **~97°→180°**
(deep load → near-straight), spine tilt **~0–25°**.

---

## 2. Pros (what's working)

- **Lower half is solid.** Every swing shows the right front-knee *pattern*: deep flex in the
  load (~97–120°) firming to a braced **148–164°** by contact. Bracing the front leg is exactly
  what MEDA's target asks for — Chae is not collapsing the front side. This is a strength to
  protect, not fix.
- **He can already produce an ideal swing.** Righty V1 and V2-seg1/seg3 land attack angle at
  **+10 to +11°** — dead center of the MLB ideal band — with **16–22°** of separation. The
  ceiling move is in his body; it just isn't repeatable yet.
- **Posture is reasonable.** Spine tilt at contact sits ~15–20° on the good swings without wild
  frame-to-frame jumps, so the rotation axis is fairly stable.

## 3. Cons (what's limiting him)

- **Inconsistent rotational separation is the headline gap.** When separation is low
  (seg0 10°, seg4 6°, seg5 11°) attack angle goes flat-to-negative (−8°, +3°, −2°) — he's
  swinging *down/early* and "spinning out" (hips and shoulders turning together). When
  separation is up (16–22°) attack angle snaps into the ideal band. **Separation is the
  variable that moves his attack angle.**
- **Attack-angle inconsistency in BP.** Of the 6 BP swings, only 2 are clearly ideal; the rest
  are flat or early. Since attack angle is also a timing metric, this is partly a timing/rhythm
  problem, not only a swing-shape problem.

## 4. Honest limitations of this read (for APE and for the roadmap)

- **The contact estimate misfires on some segments.** V2-seg2 (−169.5°) and Lefty-seg0 (−47.4°)
  are not real contact reads — the "peak hand speed" heuristic latched onto a stride/reset
  motion, not the swing. Both also show near-zero spine tilt and a near-straight knee (a
  standing frame). This is the known cost of not yet having **event detection (Phase 4)**; until
  then, trust swings where the flagged contact frame visually matches bat-ball contact.
- **2D foreshortening** understates separation in side-on views, so the *absolute* separation
  numbers are conservative — the swing-to-swing *pattern* is the trustworthy signal.
- **Lefty side is under-sampled** (one long clip, one shaky contact estimate). Need clean
  single-swing lefty clips before grading that side.

→ Handoff to **APE** ([[ape_chae]]): the lower half is good; the lever to pull is **repeatable
hip-shoulder separation**, which in this data is what carries attack angle into the ideal band.
