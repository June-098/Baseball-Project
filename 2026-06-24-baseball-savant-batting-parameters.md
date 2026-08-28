# Baseball Savant Batting Parameters — Research Report

**Date:** 2026-06-24  
**Purpose:** Identify every key batting metric on baseballsavant.mlb.com/leaderboard, explain what each measures, why it matters for biomechanical analysis, and map it to what our pipeline can compute from body keypoints.

**Primary sources:**
- [Baseball Savant CSV Documentation](https://baseballsavant.mlb.com/csv-docs)
- [MLB Statcast Glossary](https://www.mlb.com/glossary/statcast)
- [Statcast Metrics Context](https://baseballsavant.mlb.com/statcast-metrics-context)
- [MLB.com — 4 New Swing Metrics (May 2025)](https://www.mlb.com/news/new-statcast-swing-metrics-2025)
- [Driveline Baseball — Using MLB Bat Tracking Data (July 2024)](https://www.drivelinebaseball.com/2024/07/using-mlb-bat-tracking-data-to-better-understand-swings/)

---

## Overview: The Four Leaderboard Families

Baseball Savant organizes its batting leaderboards into four families, each measuring a different layer of the swing:

| Family | Leaderboard | What It Measures |
|---|---|---|
| Bat Tracking | bat-tracking, swing-path-attack-angle | The mechanics of HOW the bat moves |
| Exit Velocity & Barrels | statcast | The quality of CONTACT at impact |
| Expected Stats | expected_statistics | The likely OUTCOME of that contact |
| Batted Ball Profile | batted-ball | The TYPE and DIRECTION of contact |

The relationship between them follows a causal chain:

**Swing mechanics → Contact quality → Batted ball outcome → Actual performance**

Our pipeline sits firmly at the swing mechanics layer. The body keypoints we extract give us joint angles, velocities, and timing — the foundation from which all downstream metrics originate.

---

## PART 1 — BAT TRACKING LEADERBOARD

*Data available from 2023–present. Tracked via five high-frame-rate Hawk-Eye cameras per MLB stadium.*

---

### 1.1 Bat Speed

**Official definition (MLB Glossary):** Bat speed is measured at the sweet spot of the bat. Average bat speed is the average of the top 90% of a player's swings. League average: ~72 mph.

**Why it matters:** By physics, a faster-moving bat transfers more momentum to the ball, producing higher exit velocity. A 1 mph increase in bat speed yields roughly a 1.2 mph increase in exit velocity, all else equal. Bat speed is the single largest determinant of a batter's power ceiling.

**Biomechanical source:** Bat speed is the terminal output of the kinetic chain — hip rotation initiates torque, which transfers through the core and shoulder to the arms and wrists, which accelerate the bat. A breakdown at any segment (early hip stall, lack of shoulder turn, weak core transfer) directly caps bat speed.

**What Driveline found:** Using K-means clustering on MLB bat tracking data, hitters with high bat speed ("fast and long" cluster) included Aaron Judge, Shohei Ohtani, and Yordan Alvarez — consistent with their elite power numbers. Driveline confirmed that "a higher bat speed opens the door to a multitude of different approaches. With lower bat speed, the options are significantly lower and the margin for error is incredibly small."

**Pipeline derivability:**  
- **Direct approximation from 3D body keypoints:** Wrist velocity at contact (from keypoints_3d.json) is a reasonable proxy for bat speed, scaled by approximate bat length and grip position. Not exact, but directionally useful.  
- **Exact computation:** Requires bat endpoint tracking (Phase 2.5, deferred). Once bat tip and knob positions are tracked, speed = distance traveled by sweet spot / time elapsed.

---

### 1.2 Fast Swing Rate

**Official definition:** A fast swing is one with 75 mph or greater bat speed. Fast Swing Rate = % of total swings above this threshold.

**Why it matters:** The MLB average bat speed is ~72 mph, so 75 mph represents above-average swing velocity. Fast Swing Rate tells you how consistently a hitter generates elite swing speed, not just on cherry-picked pitches. A high fast swing rate indicates the hitter isn't selectively slow on pitches outside their hot zone.

**Pipeline derivability:**  
- Requires bat tracking. Can be estimated from wrist velocity distributions across swings within a session.

---

### 1.3 Swing Length

**Official definition (MLB Glossary):** The total (sum) distance in feet traveled by the head of the bat in X/Y/Z space, from the start of tracking data until the impact point.

**Why it matters:** Swing length is the trade-off metric in the speed/contact duality of hitting. Shorter swings are quicker to the zone and make more contact (higher batting average) but sacrifice potential power. Longer swings have more power potential but create more whiff opportunities and timing windows.

Driveline derived a kinematic estimate: using `v² = u² + 2as` (where v = bat speed, s = swing length), they computed that `time_to_contact = 2 × swing_length / bat_speed`. This means two hitters with the same bat speed can have very different time-to-contact profiles depending on swing length — which directly affects how early the hitter must begin their swing decision.

**What Driveline found:** When clustering MLB hitters by bat speed + swing length alone, they produced four natural groups: fast/long (Judge, Ohtani), slow/short (contact hitters), average/short (efficient hitters), average/long (struggling hitters). The "average/short" cluster (Juan Soto, Bobby Witt Jr., Gunnar Henderson) had the highest wOBA, confirming that efficiency to the ball matters more than raw length.

**Pipeline derivability:**  
- Requires bat tip tracking (Phase 2.5). Can be estimated from wrist path length as a lower-bound proxy.

---

### 1.4 Attack Angle

**Official definition (MLB Glossary):** The vertical angle at which the sweet spot of the bat is traveling at the point of impact with the ball. A 0° would be perfectly flat; positive = bat moving upward, negative = bat moving downward.

**Official definition — Ideal Attack Angle:** A batted ball hit with an attack angle between 5° and 20°.

**Why it matters:** This is arguably the most important single metric for batting development and the one most directly tied to swing timing.

From the official MLB May 2025 explainer: "Attack angle is, among other things, a timing metric... you might get dozens of attack angles during the swing — 4 degrees per frame — and the one that matters is the one that happens at contact. That means that having an undesirable attack angle might be about being early or late, as well as the way you're moving your bat."

The numbers from Baseball Savant's metrics context page (post-2023 All-Star Game data) confirm the performance gap:

| Attack Angle Category | AVG | SLG | wOBA |
|---|---|---|---|
| Ideal (5°–20°) | .272 | .487 | .323 |
| All others | .250 | .354 | .261 |

That's a 62-point wOBA differential between ideal and non-ideal attack angles — a massive gap. The ideal range matches the natural downward trajectory of an incoming pitch (-5° to -20°), meaning the batter is literally matching the ball's plane. Ted Williams wrote about this concept in *The Science of Hitting* (1971): "I advocate a slight upswing (from level to about 10 degrees)."

**Real example — Corbin Carroll (Arizona):** His ideal attack angle was in the 25th percentile in the first half of 2024 (.213 AVG). After adjustments, it reached the 100th percentile in 2025 (best in MLB), correlating with an NL MVP-caliber season.

**Pipeline derivability:**  
- **Best approximation from body keypoints:** Wrist trajectory angle at the moment of estimated contact. If the wrist is moving at +12° at contact time (detected via Phase 4 event detection), that's a strong proxy for attack angle.
- **Exact computation:** Requires bat sweet spot position over time (Phase 2.5).

---

### 1.5 Swing Path Tilt

**Official definition (MLB Glossary):** The vertical angular orientation of the "plane" of the swing, as compared to the ground, defined by the path of the bat in the 40 ms prior to contact.

Think of it as the overall shape of the swing: an uppercut has high tilt (steep, 40°+), a flat contact hitter has low tilt (~20°). MLB average: 32°.

**Why it matters:** Unlike attack angle (which measures what's happening *at* contact), swing path tilt tells you about the overall arc geometry. The two work together: you can have a steep swing path but still arrive at ideal attack angle if your timing is perfect. The MLB article describes it as a stylistic descriptor rather than a "more is better" metric — though extreme flat or steep swings tend to correlate with either low power or high whiff rates respectively.

Notable tilt values from 2025:
- Riley Greene: 46° (most extreme uppercut)
- Freddie Freeman: 42° (large uppercut, productive)
- Yandy Diaz: 24° (flattest, contact-focused)
- Jake Burger: 24° (flat, inconsistent results)

**Pipeline derivability:**  
- Approximable from wrist/elbow path geometry over the 40ms prior to estimated contact.
- Exact: requires bat head path tracking.

---

### 1.6 Attack Direction

**Official definition (CSV docs):** The horizontal angle of the direction in which the sweet spot of the bat is traveling at impact, compared to a line from home plate to straightaway center field. Expressed as pull (positive) or oppo (negative).

**Why it matters:** This reveals horizontal timing and pull tendency. A pull-oriented attack direction (bat still moving left-to-right for right-handers) correlates with pulling for power; oppo-oriented indicates the hitter is letting the ball travel deeper. From the MLB article: pull-side hitters have approximately 120 extra points of slugging versus oppo-oriented hitters.

**Pipeline derivability:**  
- Approximable from wrist horizontal velocity vector at contact. If the wrist is tracking toward the pull-side at contact, attack direction will be pull-oriented.

---

### 1.7 Squared-Up Rate

**Official definition (MLB Glossary):** A swing where at least 80% of the potential exit velocity (based on swing speed and pitch speed) is converted into actual exit velocity. Formula: `actual_EV / (bat_speed + pitch_speed × 0.2) ≥ 0.80`.

**Why it matters:** Squared-up rate separates bat-to-ball skill from raw power. A 75 mph swing on a 95 mph fastball has a theoretical maximum exit velocity of roughly 105 mph. If the actual exit velocity is 84+ mph, that swing was squared up. It measures contact efficiency, independent of bat speed. Driveline notes that squared-up rate "is incredibly similar to the smash factor metric we have been using to quantify bat-to-ball skills for years."

**Pipeline derivability:**  
- Requires both bat speed and actual exit velocity — needs bat tracking + ball exit data. Not directly computable from body keypoints alone.

---

### 1.8 Blasts

**Official definition (MLB Glossary):** A more valuable subset of squared-up balls. A blast is a batted ball that was both squared-up AND had a fast swing (≥75 mph bat speed).

**Why it matters:** Blasts represent the best possible swing outcome — elite speed *and* elite contact efficiency simultaneously. A hitter with high blast rate is consistently generating elite contact. Per the metrics context page, blasts show the highest correlation with hard contact and home run production of any single swing event.

Blasts separate elite hitters from merely strong ones: you can have a fast swing with poor contact (whiffs), or good contact with a slow swing (weak grounders). A blast requires both.

**Pipeline derivability:**  
- Composite of bat speed + squared-up, both requiring bat tracking. Not derivable from body keypoints alone.

---

### 1.9 Swords

**Official definition (MLB Glossary):** A bat tracking metric that quantifies when a pitcher forces a batter to take a non-competitive, ugly-looking swing. Named after the image of a batter "falling on their sword."

**Why it matters:** Useful as a pitcher-side metric. For batting analysis, a hitter with high sword rate is being exploited by pitch sequences or locations that disrupt their timing — visible in the body as awkward unloading, off-balance completion, or rushed hip rotation.

**Pipeline derivability:**  
- Qualitatively derivable from our 3D skeleton: a sword will show as poor hip rotation completion, excessive spine lean, and/or arm-only swing with no rotational contribution.

---

## PART 2 — EXIT VELOCITY & BARRELS LEADERBOARD

*Available from 2015–present.*

---

### 2.1 Exit Velocity (EV)

**Official definition:** How fast, in miles per hour, a ball was hit by a batter.

**Why it matters:** Exit velocity is the most direct output of contact quality. All other things equal, harder contact travels farther and reaches fielders faster. The correlation between average EV and offensive production is well established — it's the best single predictor of power.

**Biomechanical source:** EV is determined by `collision physics = bat mass × bat speed + ball mass × pitch speed`, weighted by contact quality (squared-up factor). The body contributes through every segment of the kinetic chain that produces bat speed.

**Pipeline derivability:**  
- Requires ball tracking. Not computable from body keypoints or bat tracking alone without additional sensors or radar.

---

### 2.2 EV50

**Official definition:** For a batter, EV50 is the average exit velocity of the hardest 50% of their batted balls. (For pitchers, it's the softest 50% allowed.)

**Why it matters:** Regular average EV is dragged down by mis-hits and soft contact. EV50 gives a cleaner ceiling picture of a hitter's top-end power by ignoring the worst half of their contact.

**Pipeline derivability:**  
- Requires ball tracking.

---

### 2.3 Adjusted EV (Hyper Speed)

**Official definition (CSV field `hyper_speed`):** Adjusted Exit Velocity sets every batted ball hit below 88 mph as 88 mph; otherwise uses actual EV. The 88 mph floor is the threshold below which batted balls almost never become hits regardless of angle.

**Why it matters:** By flooring at 88 mph, this metric removes the "noise" of soft contact that all hitters occasionally produce and gives a better representation of average productive contact. Developed by Tom Tango (MLB's chief statistician).

**Pipeline derivability:**  
- Requires ball tracking.

---

### 2.4 Barrel

**Official definition (MLB Glossary):** A batted ball with the perfect combination of exit velocity and launch angle. Specifically: exit velocity ≥ 98 mph AND launch angle between 26°–30°. At lower exit velocities (98 mph requires 26–30°; 99 mph expands the window; by 116 mph+ any launch angle 8–50° qualifies).

**Why it matters:** Barrel rate is the strongest predictive Statcast metric for home run rate, ISO (isolated power), and future offensive production. Research consistently shows barrel rate outperforms simple hard-hit rate as a predictor because it adds the angle component — a 100 mph grounder and a 100 mph rocket at 28° are not equivalent, but hard-hit rate treats them identically.

**Pipeline derivability:**  
- Requires both exit velocity and launch angle — both require ball tracking.
- **However:** our pipeline can predict barrel *likelihood*. A swing that produces ideal attack angle (5–20°), from a high-speed hip rotation, with full extension at contact, has a much higher probability of producing a barrel. We can create a "predicted barrel probability" score from body mechanics alone.

---

### 2.5 Hard-Hit Rate

**Official definition:** A batted ball hit with exit velocity of 95 mph or greater. Hard-Hit Rate = % of batted balls exceeding this threshold.

**Why it matters:** Hard-hit rate is the broadest quality-of-contact filter. It tells you how often a hitter is generating significant force at impact. Unlike barrel, it doesn't penalize for launch angle — a 96 mph grounder counts. This makes it less predictive of outcomes but more stable as a contact quality signal.

**Pipeline derivability:**  
- Requires ball tracking.

---

### 2.6 Launch Angle Sweet-Spot %

**Official definition:** A batted ball hit with launch angle between 8° and 32°. This range corresponds to line drives and fly balls that regularly become hits or extra-base hits; grounders and popups are below and above this range respectively.

**Why it matters:** Sweet-spot % is a batted-ball outcome proxy that requires no additional tracking beyond launch angle — useful because it maps directly to the attack angle (pre-contact) we *can* compute. A batter whose attack angle is consistently in the 5–20° ideal range (body mechanics) will produce significantly more balls in the 8–32° sweet spot (batted ball outcome).

**Pipeline derivability:**  
- Not computable without ball exit angle. But the correlation to attack angle means our pipeline can *predict* sweet-spot likelihood from swing mechanics.

---

## PART 3 — EXPECTED STATISTICS LEADERBOARD

*Available from 2015–present. Model trained on all Statcast batted ball data.*

---

### 3.1 Expected Batting Average (xBA)

**Official definition (MLB Glossary):** xBA measures the likelihood that a batted ball will become a hit. Each batted ball is assigned an xBA based on how often comparable balls — in terms of exit velocity, launch angle, and sprint speed — have become hits since 2015.

**Why it matters:** xBA removes the luck variable from batting average. A batter hitting .220 with an xBA of .280 is likely running into bad luck (poorly timed defensive plays, opposite field grounders); regression to xBA is expected over larger samples. For player development, xBA is more useful than actual AVG because it evaluates the quality of contact, not what happened to it afterward.

**Pipeline derivability:**  
- Requires EV + LA + sprint speed. Our pipeline can contribute the underlying swing quality that produces the EV/LA inputs, but xBA itself requires ball tracking data.

---

### 3.2 Expected Weighted On-Base Average (xwOBA)

**Official definition (MLB Glossary):** xwOBA is formulated using exit velocity, launch angle, and on certain batted balls, sprint speed. Every batted ball is assigned a single/double/triple/HR probability based on comparable balls since 2015.

**Why it matters:** xwOBA is the gold-standard overall hitting quality metric. Regular wOBA is contaminated by luck in ball placement; xwOBA removes that and evaluates how well you actually hit the ball. A large xwOBA–wOBA gap indicates a hitter either getting consistently lucky (likely to regress) or unlucky (likely to improve).

From the MLB glossary: "xwOBA and all expected stats are more nuanced than normal versions and remove the effect of luck on player evaluation — a player with a high xwOBA but a low actual wOBA might be experiencing bad luck."

**Pipeline derivability:**  
- Not computable directly. But our pipeline is building the mechanical foundation that drives xwOBA outcomes.

---

### 3.3 Expected Slugging (xSLG)

**Official definition:** xSLG estimates what a batter's slugging percentage should be based on the quality of their contact (EV + LA), independent of what actually happened with each ball in play.

**Why it matters:** Similar reasoning to xwOBA — isolates contact quality from luck. Particularly useful for evaluating power hitters.

---

## PART 4 — BATTED BALL PROFILE LEADERBOARD

---

### 4.1 Batted Ball Types

**Official definition (CSV field `bb_type`):** Four categories — `ground_ball`, `line_drive`, `fly_ball`, `popup`. Defined by launch angle buckets:
- Ground ball: LA < 10°
- Line drive: 10° ≤ LA < 25°
- Fly ball: 25° ≤ LA < 50°
- Pop up: LA ≥ 50°

**Why it matters:** The distribution of batted ball types is a fingerprint of a hitter's approach and mechanics. Pure pull-side fly ball hitters will show high FB%, high HR, but also high K%. Contact-first hitters have high LD% and GB% with lower HR. For our purpose: a batter who consistently rolls over (ground balls to the pull side) has a mechanical issue detectable in their hip rotation timing and attack angle.

**Pipeline derivability:**  
- Predictable from attack angle and attack direction. Not exactly computable without ball exit data, but approximate from swing mechanics.

---

## PART 5 — PIPELINE DERIVABILITY MATRIX

This is the core of why Baseball Savant matters to our project. The table below maps every key metric to what we can derive at each phase of our pipeline.

| Metric | From 3D Keypoints (Phase 3) | + Bat Tracking (Phase 2.5) | + Ball Tracking |
|---|---|---|---|
| **Hip rotation velocity** | ✅ Direct | — | — |
| **Hip-shoulder separation** | ✅ Direct | — | — |
| **Shoulder rotation angle** | ✅ Direct | — | — |
| **Extension at contact** | ✅ Direct (elbow angle) | — | — |
| **Weight transfer** | ✅ Direct (hip displacement) | — | — |
| **Time to contact** | ✅ With event detection (Phase 4) | — | — |
| **Swing timing (early/late)** | ✅ Via attack angle proxy | — | — |
| **Attack angle (approx.)** | ✅ Wrist trajectory angle | ✅ Exact | — |
| **Ideal attack angle %** | ≈ Approx via wrist | ✅ Exact | — |
| **Attack direction** | ≈ Wrist horizontal vector | ✅ Exact | — |
| **Swing path tilt** | ≈ Approx via elbow/wrist arc | ✅ Exact | — |
| **Bat speed** | ≈ Wrist velocity (rough proxy) | ✅ Exact | — |
| **Swing length** | ≈ Wrist path length (lower bound) | ✅ Exact | — |
| **Squared-up rate** | ❌ | ✅ Needs EV too | ✅ |
| **Blasts** | ❌ | ✅ Needs EV too | ✅ |
| **Fast swing rate** | ❌ | ✅ Exact | — |
| **Exit velocity** | ❌ | ❌ | ✅ |
| **Launch angle** | ❌ | ❌ | ✅ |
| **Barrel rate** | ❌ | ❌ | ✅ |
| **Hard-hit rate** | ❌ | ❌ | ✅ |
| **xBA / xwOBA / xSLG** | ❌ | ❌ | ✅ |
| **Batted ball type** | ≈ Predictable from swing | ❌ | ✅ |

**Legend:**  
✅ = Directly computable  
≈ = Approximable / proxy metric  
❌ = Not computable at this layer

---

## PART 6 — WHICH METRICS TO PRIORITIZE IN PHASE 5

Given our current pipeline (body 3D keypoints → event detection → metrics), Phase 5 should target the following in priority order:

**Tier A — Directly computable from body keypoints + event detection:**

1. **Hip rotation velocity** (deg/sec from toe touch to contact) — The most important physical driver of bat speed. Coaches refer to this as "hip speed into the ball."
2. **Hip-shoulder separation** (angle delta between hip plane and shoulder plane at launch point) — Elite hitters show 30–45° separation; this creates torque that transfers into bat speed. Ted Williams described this as "clearing the hips first."
3. **Shoulder tilt at contact** (vertical angle of shoulder line) — Should be slightly negative (front shoulder higher) for optimal launch angle influence.
4. **Extension score** (elbow extension at contact — degree to which both arms are extended) — Full extension at contact maximizes leverage arm and is associated with higher exit velocity.
5. **Time to contact** (from event-detected launch frame to contact frame) — Encodes decision time and swing efficiency.
6. **Knee flex during load** (front knee angle at heel plant vs. contact) — Drive leg mechanics.
7. **Posture / spine angle** (trunk lean consistency through swing) — Excessive lean or drift disrupts rotation mechanics.

**Tier B — Approximable from body keypoints (less precise but useful):**

8. **Estimated attack angle** (wrist vertical velocity direction at contact) — Strong proxy even without bat tip.
9. **Attack direction** (wrist horizontal vector at contact) — Pull vs. oppo tendency from body mechanics.
10. **Weight transfer** (hip center horizontal displacement from load to contact) — "Getting into the ball" vs. drifting away.

**Tier C — Requires bat tracking to be exact (Phase 2.5):**

11. Bat speed, swing length, swing path tilt, exact attack angle, squared-up rate, blasts.

---

## PART 7 — HOW THE METRICS CONNECT TO FEEDBACK FOR HIGH SCHOOL ATHLETES

This is the product layer — translating technical metrics into actionable swing cues for the target user (high school athlete).

| What We Compute | What It Tells the Athlete |
|---|---|
| Hip rotation velocity < threshold | "Your hips are stalling. Drive your back hip harder toward the pitcher." |
| Hip-shoulder separation < 25° | "You're spinning out too early. Your hands are leading instead of your hips." |
| Extension score < 80% | "You're casting away at contact. Stay inside the ball and extend through it." |
| Attack angle < 5° at contact | "You're hitting down on the ball. You're either early or swinging too flat." |
| Attack angle > 25° at contact | "You're late. You're catching the ball too deep and rolling under it." |
| Low weight transfer | "You're staying back too long. Get your weight into the ball." |

The loop closes back to Baseball Savant: if we can estimate attack angle and hip metrics, we can tell the athlete exactly what Statcast would show — and what to fix — without them ever needing a stadium's worth of Hawk-Eye cameras.

---

## Key Reference Numbers (from official sources)

| Metric | Poor | Average | Elite |
|---|---|---|---|
| Bat Speed | < 65 mph | ~72 mph | 80+ mph |
| Swing Path Tilt | < 20° or > 50° | ~32° | Context-dependent |
| Attack Angle | < 0° or > 25° | ~10° | 5–20° (ideal) |
| Ideal Attack Angle % | < 40% | ~50% | 65–74% (top tier) |
| Barrel Rate | < 5% | ~7% | 15%+ |
| Hard-Hit Rate | < 30% | ~36% | 50%+ |
| EV (Average) | < 85 mph | ~88 mph | 95+ mph |

---

*Sources: baseballsavant.mlb.com (official), mlb.com/glossary/statcast, mlb.com/news/new-statcast-swing-metrics-2025, drivelinebaseball.com*
