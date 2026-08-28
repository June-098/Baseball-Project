

- Researched what kind of AI tools do professional and amateur baseball players use for batting. For professional players, they use [Hawk-EyeInnovations](https://www.hawkeyeinnovations.com/sports/baseball), [Driveline baseball](https://www.drivelinebaseball.com/), and [Statcast](https://baseballsavant.mlb.com/). Statcast is more of statistical analysis for professional players, so this can be reviewed after I finish batting and pitching AI assistant. For driveline baseball, hitting is seperated into pro, college, and high school. They divided into three skill groups, but my focus will be high school and below for the phase 1. I should think about how I can approach to college and professional when I deploy for high schoolers. For the hawk-eye innovation, tool tracks **every pitch and batted ball** in 3D (velocity, spin, release point, launch angle). It feeds AI-driven models for pitch shape optimization, swing plane & contact quality, and predictive outcomes. Players use it to compare actual vs optimal launch windows and pitchers redesign arsenals.
- Researched what amateur athletes use for baseball AI tools. There are [HitTrax](https://www.hittrax.com/), [Rapsodo](https://rapsodo.com/), and [DiamondKinetics](https://www.diamondkinetics.com/). HitTrax does radar+camera+ AI simulation where it measures exit velo, launch angel, and spray charts. Amateurs use it to get immediate feedback after swing, and compare sessions week to week. In reality, athletes must go to training facility and its still hardware dependent. Rapsodo does portable radar + AI models for pitching (velocity, spin rate, axis) and hitting (exit velo, launch angle, and distance). Rapsodo is also hardware dependent, but widely used in elite high school programs. Diamond Kinetics is wearable device where its mounted on bat using IMU and AI analysis. This measures bat speed, attack angle, and time to contact. It is cheaper then other hardware, but it limits vision data (no ball flight) and great for mechanics but there is no outcomes. 
#SportFX
- The idea of me uploading videos to the tool and AI will analyze the mechanices and provide feedback to user is accomplished in [sportfx.ai](https://sportfx.ai/pricing). Pricing is broken down into $15.95 per month, $29.95 per month, and 99.95 per month. Each price can upload 10, 30, and 250 videos per month. This tool is for atheletes who wants instant feedback, tracking your progress, personalized drills, compare to the pros, earn achievements, and make training fun. SportFX partnered with NVidia. SportFX uses advanced computer vision and machine learning to analyze athletic movements with unprecedented accuracy by acheiving 3D motion capture (full body tracking from 2D video), AI analysis Engine (trained on pro-level mechanics), and instance feedback.
- This tool uses 2D to 3D (monocular 3D lifting). It is a computer vision term for techiniques used to estimate 3D information from a single 2D image (monocular vision).
	- 3D pose model trained on large motion datasets
	- Temporal modeling (sequence-based) to reduce jitter
	- Camera normalizaiton/implicit calibration assumptions
- To deliver "baseball feedback", I must detect key events:
	- toe touch, heel plant, swing launch, contact, extension

Action Items:
- [ ] Pick my narrow MVP that I can win
- [ ] Decide my "truth source for credibility"
- [ ] Build the pipeline before UI
	- [ ] Upload + storage + privacy
	- [ ] Pose -> 3D -> smoothing
	- [ ] Event detection
	- [ ] 5-10 metrics that coaches agree matter
	- [ ] Feedback + drill mapping
	- [ ] Progress tracking

---

## 2026-06-09 — CV/ML Stack Research & Architecture Decision

### Research Reviewed
- Markerless Motion Capture (Theia3D)
- ByteTrack Multi-Object Tracking (MOT)
- Competitive landscape (Hawk-Eye, Driveline, HitTrax, Rapsodo, DiamondKinetics, SportFX)

### Key Findings

**ByteTrack Assessment — Not the primary tool**
ByteTrack is a multi-object tracker that outputs bounding boxes + persistent IDs across frames. It does not produce joint keypoints or biomechanical data. For a single-player batting/pitching analysis app, it adds no analysis value. Its only applicable role is isolating a player from a crowded frame as a pre-processing step. It should not be treated as the core CV library for this project.

**Theia3D (Markerless Motion Capture) Assessment — Gold standard, not shippable to consumers**
Theia3D is the benchmark: 124 anatomical landmarks, synchronized multi-camera array, full body + bat + ball tracking in one coordinate system. Requires NVIDIA GPU desktop, multi-camera lab rig, and operator calibration. Not viable for a consumer phone app. Useful as: (1) training data source, (2) benchmark to validate our own model outputs, (3) aspirational accuracy target for a later hardware-optional premium tier.

### Decided Architecture — Single-Camera Consumer Pipeline

| Layer | Tool | Notes |
|---|---|---|
| Pose estimation (server) | YOLOv8-Pose | High accuracy on uploaded video |
| Pose estimation (on-device) | MediaPipe Pose | Real-time, runs on phone, 33 landmarks |
| 2D → 3D Lifting | MotionBERT | SOTA single-camera 3D pose, server-side |
| Ball tracking | TrackNet | Purpose-built for fast-moving sport balls |
| Event detection | Custom LSTM/Transformer | Swing phase segmentation over joint time series |
| Training annotation | CVAT | Label baseball-specific training footage |
| Backend API | FastAPI (Python) | Async, ML-friendly |
| GPU inference | AWS EC2 g4dn or Modal.com | On-demand GPU, cost-effective |
| Mobile + Web | React Native | Single codebase for iOS, Android, Web |
| Video storage | AWS S3 + CloudFront | Scalable, standard |
| Training data source | Statcast + YouTube pro footage | Ground truth for fine-tuning |

### Why This Stack Wins
- MediaPipe + YOLOv8-Pose run from a single phone camera — no hardware required
- MotionBERT handles the monocular 3D lifting that SportFX uses (confirmed competitor approach)
- TrackNet solves the fast-ball detection problem standard YOLO detectors fail on
- CVAT handles labeling of baseball-specific training data
- React Native delivers iOS + Android + Web from one codebase

### Competitive Positioning vs. SportFX
SportFX uses the same monocular 3D lifting approach. Differentiation must come from: (1) baseball-specific model fine-tuning on pro player data, (2) coaching-grade metric depth — not just swing path, but hip-shoulder separation and kinetic chain sequencing, (3) Phase 1 focus on high school athletes that SportFX doesn't specifically target.

### Updated Action Items
- [ ] **MVP scope decision** — Batting only for Phase 1 (high school); pitching in Phase 2
- [ ] **Truth source** — Choose credibility anchor: Driveline partnership, certified coach advisory board, or Statcast metric alignment
- [ ] **Pipeline build order:**
  - [ ] 1. Video upload + S3 storage + basic privacy (blur faces of bystanders)
  - [ ] 2. YOLOv8-Pose → MotionBERT → 3D joint output from test videos
  - [ ] 3. TrackNet ball tracking on batting clips
  - [ ] 4. Event detection (toe touch, heel plant, launch, contact, extension)
  - [ ] 5. Compute 5–7 batting metrics coaches validate (attack angle, hip rotation, shoulder separation, bat speed, extension)
  - [ ] 6. Feedback + drill mapping engine
  - [ ] 7. Progress tracking across sessions
  - [ ] 8. React Native app (after pipeline proven)
- [ ] **Research next:** MotionBERT fine-tuning on baseball-specific pose datasets; TrackNet baseball adaptation examples


Create a workstation called Personal Career Project following the workstation creation instructions in my root CLAUDE.md. Then use all md files I've dropped into this folder to use as a reference, background, and research topics. Goal:

- goal of this project is to create resume, preparation md files for behavioral and technical questions based on the job description. However, each Job application/description should create another sub folder. For an example, If I have SR Engineer for Google, and Engineer for Amazon, create subfolder for each with title and company.
- I have included several resumes that I used and got into the interview process. Please use this as a reference resume in order to grab important projects and classes for a new resume.
- Preparation md files should be generated when I get into the interview process. I have included my past behavioral preparation in the past so you can use this as a reference to create a new preparation file. Don't run preparation with the resume maker. 
- Referring to the Notes.md, please use this to update our progress and findings that we have done. 