# APE — Athlete Performance Enhancement
## Subject: Chae (friend) — ranked corrections

**Agent:** APE (step 3 of MEDA → APA → APE)
**Run date:** 2026-06-24
**Inputs:** [[meda_2d_target]] (the target) + [[apa_chae]] (the current read).
**Rule:** every correction names *why* it follows from this athlete's MEDA target and APA gap —
no generic advice.

---

## The one-sentence diagnosis

Chae's lower half already does its job (front-knee brace 148–164° at contact, exactly on
target), so the lever that will move his hitting most is **repeatable hip-shoulder separation**
— the single variable that, in his own data, carries attack angle into the ideal 5–20° band.

---

## Ranked recommendations

### 1. Build repeatable hip-shoulder separation *(highest priority)*
- **Why (from the data):** MEDA target is ~25–45° at launch. APA shows the direct link in Chae's
  own swings — sep **16–22°** → attack angle **+10 to +11°** (ideal); sep **≤10°** → attack angle
  **−8° to +3°** (flat/early). Separation is the cause, attack angle is the effect. He's
  "spinning out" (hips and shoulders turning together) on the misses.
- **Cue / drill:** "hips first, hands back" — start the hips while keeping the hands and front
  shoulder closed a beat longer. Step-back or hook-em (separation) drills; feel the stretch
  across the front of the torso before the hands fire.
- **Confirm on the overlay:** the cyan shoulder line and magenta hip line should *split* before
  the hands go — watch the **Hip-Shoulder Sep** number climb past ~20° and turn green.

### 2. Make the ideal attack angle repeatable *(timing)*
- **Why:** APA — only 2 of 6 BP swings hit the ideal band; the rest were flat/early. Attack angle
  is also a *timing* metric (early/late changes the angle at contact), so this is rhythm, not a
  new swing shape — he already produces +10° when timed up.
- **Cue / drill:** tee at mid-thigh, groove the feeling of the hands working slightly **up**
  through the ball; front-toss with a consistent rhythm/load trigger so contact timing repeats.
- **Confirm on the overlay:** **Attack Angle** at the CONTACT flag sitting green (+5 to +20°)
  across *most* swings, not just the best one.

### 3. Protect the front-leg brace *(don't break what works)*
- **Why:** APA — the brace pattern (deep load → ~150–165° at contact) is already a strength.
  Chasing separation can tempt a hitter to drift onto a soft front leg. Keep it firm.
- **Confirm on the overlay:** **Front-Knee Angle** should still *firm up* toward contact while
  he works on #1.

### 4. Gather clean lefty clips before prescribing the lefty side
- **Why:** APA — the only lefty read came from one long clip with an unreliable contact estimate
  (−47.4°). No trustworthy lefty baseline yet.
- **Action:** record 3–5 single-swing lefty clips, re-run `--stage apply2d`.

---

## Caveats carried forward (don't over-coach off bad frames)

- Two segments (V2-seg2, Lefty-seg0) had **mis-estimated contact frames** (peak-speed heuristic
  caught a stride/reset). Only coach off swings where the on-screen CONTACT flag matches visible
  bat-ball contact. Real **event detection (Phase 4)** will remove this caveat.
- All numbers are **2D view-dependent proxies**; the trustworthy signal is the *pattern across
  swings*, not any single absolute value. Exact attack angle/bat speed wait on bat tracking
  (Phase 2.5, deferred).

---

## Product note (translating to the athlete)

For a high-school hitter, the whole report reduces to one screen cue:
**"Split your hips and shoulders sooner — when that number turns green, your swing plane does
too."** Everything else (knee, posture) is already in range.
