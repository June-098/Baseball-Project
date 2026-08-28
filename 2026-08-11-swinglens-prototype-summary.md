# SwingLens Prototype: Repo Summary and Run Instructions

**Source:** https://github.com/memekr/swinglens-prototype (cloned into `src/swinglens-prototype/` on 2026-08-11)
**Production:** https://swinglens-prototype.vercel.app

## What it is

SwingLens is a working prototype of a phone web app (a PWA, an installable progressive web app that behaves like an app but is really a website). It reviews a baseball swing video entirely inside the phone's browser. No video ever leaves the phone: there's no upload, no backend server, no API. All pose detection runs on-device using Google's MediaPipe library.

## How it relates to our pipeline

This is a separate track from the `src/` pipeline built from `YOLO_Baseball.ipynb` (YOLO26 pose, then MotionBERT for 3D lifting, run server-side in Colab). SwingLens runs a lighter on-device 2D pose model (MediaPipe Pose Landmarker Lite, 33 landmarks) directly in the browser. It does not attempt 3D, bat speed, exit velocity, or true attack angle (see "What it does not measure" below). Treat it as a reference for the on-device, PWA product direction and for its competitor research, not as a replacement for the 3D metrics work already in progress.

## Tech stack

| Layer | Choice |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack), React 19, TypeScript |
| Pose model | MediaPipe Tasks Vision 1.0.1, Pose Landmarker Lite, bundled locally instead of fetched from a CDN |
| Hosting | Vercel, static only, no server or API routes |
| Testing | Vitest for unit tests, Playwright for mobile end-to-end tests |
| Package manager | npm |

## Repo layout

- `app/`: the single page (`page.tsx`), layout, and PWA manifest
- `components/`: `SwingAnalyzer` (record or upload), `ReviewStudio` (the "Frame Lab" scrubber), `AnalysisReport` (Player and Coach report views), `SkeletonView` (pose overlay)
- `lib/`: the real logic. `pose-engine.ts` (MediaPipe wrapper), `video-analysis.ts` (frame sampling), `analysis.ts` (checkpoint detection and scoring), `geometry.ts` (angle math), `drills.ts` (practice card mapping), `demo.ts` (synthetic demo data for testers without a video)
- `public/models/`, `public/wasm/`: the bundled pose model (5.5 MB) and MediaPipe WASM runtime (about 32 MB total), checked directly into the repo so the deployed app never depends on an external CDN at inference time
- `docs/architecture.md` and `docs/competitive-analysis.md`: worth reading directly. The second is a feature-by-feature comparison against Onform, Kinovea, Sports2D, Mustard, and six other competitor products, with an explicit "added or not added, and why" column for each

## What it measures

*Updated 2026-08-13, see the Local changes section below.*

- On-device pose detection (MediaPipe, 33 raw points), mapped to this project's own 17-point body map (`Baseball Resources/body-labeling.md`)
- Four swing checkpoints, in the batting sequence the coaching team defined: **Trigger** (first move that sets rhythm/timing), **Execution** (front-foot plant, then the kinetic chain), **Impact** (peak hand speed, the bat-ball candidate), **Follow-through** (the finish, plus a tracked hand-path trajectory)
- 2D cues: lead-knee angle, torso lean, shoulder-hip line, relative head travel, and (Follow-through only) swing path shape
- A mechanics score, withheld when video quality or framing is too weak for the pose data to be trustworthy

## What it does not measure

Straight from the repo's own docs, worth keeping in mind before showing this to anyone. It does not measure actual bat-ball contact time, bat speed, exit velocity, or true (3D) attack angle. "Contact" is a proxy: the highest observed wrist speed, not a detected ball strike.

## How to run it

Verified on this machine today (Node v26.7.0, npm 11.19.0):

```bash
cd "Personal Baseball Project/src/swinglens-prototype"
npm install
npm run dev
```

Then open **http://localhost:3000**. Camera recording only works over HTTPS or on `localhost`. That's a browser security rule, not a bug: `localhost` is fine for local dev, but a phone on the same Wi-Fi hitting your laptop's IP address will not get camera access without HTTPS.

To validate before trusting a change, all pass as of today:

```bash
npm run lint      # clean
npm test          # 9/9 unit tests pass (analysis, drills, geometry)
npm run build     # production build succeeds, fully static, 3 routes
```

Optional and heavier, end-to-end mobile tests (not run today, needs browser downloads first):

```bash
npx playwright install chromium webkit
npm run test:e2e
```

## Where it runs

- **Local dev:** `localhost:3000`, on any machine with Node installed. No `.env` file, no API keys, no database. The app is entirely client-side.
- **Production:** already deployed at https://swinglens-prototype.vercel.app. Vercel hosts the static build, so there is no server to manage.
- **On a phone:** open the Vercel URL (or your own deploy) in mobile Safari or Chrome. It installs as a PWA (add to home screen) and works offline after the first load, since the model and WASM runtime are cached locally.

## Access note

The GitHub repo was private and this account had no access to it until 2026-08-11, when the owner (`memekr`) made it public so it could be cloned here.

## Local changes (2026-08-13, not yet pushed)

Two changes were made in this local clone, verified with the same lint/test/build/dev commands above:

1. **Body labeling:** every landmark-consuming file (`lib/skeleton.ts` new, plus `geometry.ts`, `analysis.ts`, `pose-engine.ts`, `demo.ts`, `SkeletonView.tsx`) now reads joints by name from this project's own 17-point body map instead of raw MediaPipe index numbers. Rendering now draws all 15 named points (head, torso, center hip, and left/right shoulder, elbow, hand, hip-joint, knee, foot) with a proper spine line, instead of the prior 13-point box outline.
2. **Phase model:** setup/launch/contact/follow was replaced with Trigger → Execution → Impact → Follow-through, matching the biomechanical sequence the user specified. Execution now anchors on a real front-foot-plant detection instead of a fixed time offset. Follow-through gained a tracked hand-path visualization (polyline drawn on the frame image) and a fifth "swing path shape" metric, checked against a circular, slightly-upward reference shape.

**Not pushed:** this account has read access to `memekr/swinglens-prototype` (temporarily public) but no write access, and the live Vercel deploy is on the repo owner's account, so `swinglens-prototype.vercel.app` still shows the old version. To try the changes, run locally (steps above) — the demo report ("Explore sample report" on the homepage) exercises all of it without needing a real video.
