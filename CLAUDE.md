# Personal Baseball Project

## Identity

This workstation is for building a baseball video analysis app, a tool where athletes upload batting or pitching footage and receive AI-generated feedback on mechanics and performance metrics (swing path, attack angle, bat speed, exit velocity, hip-shoulder separation, and more). It routes all work related to: ML pipeline development, computer vision research, app architecture decisions, competitive analysis, training data strategy, and product roadmap. It does not route general productivity tasks, email, or finance work.

## Resources

|Resource|Read when...|
|:--|:--|
|Baseball Resources/markerless-motion-capture.md|Researching pose estimation approaches or multi-camera systems|
|Baseball Resources/bytetrack-mot.md|Reviewing multi-object tracking options|
|Baseball Resources/body-labeling.md|Working on skeleton keypoint definitions or annotation schema|
|Baseball Resources/notes.md|Reviewing full research history and prior findings|
|Baseball Resources/RAG Resources/|Building or querying the coaching RAG index|
|Baseball Resources/golden_questions.jsonl|Measuring retrieval with rag_eval (not model prompts)|

## Workflow

1. **Research phase**. Study competitor tools, CV libraries, and ML models. Document findings in notes with today's date as section header.
2. **Architecture decisions**. Evaluate options and lock down the stack. Record final decisions and rationale in MEMORY.md under Key Decisions.
3. **Pipeline build**. Build in order: upload → pose → 3D lifting → event detection → metrics → feedback engine → UI. The feedback engine is the Agent Team below (MEDA → APA → APE, run in sequence).
4. **Validation**. Test each pipeline layer against pro player footage before moving to the next layer.
5. **Product**. Build React Native app only after pipeline produces reliable metrics.

## Output Conventions

- **Any code that outputs a video saves the MP4 to the `Batting Diagnoses` folder**
  (`G:/My Drive/Baseball Project/Batting Diagnoses/` = Colab `/content/drive/My Drive/Baseball Project/Batting Diagnoses/`).
  Applies to all current and future code (2D overlays, 3D renders, bat tracking, etc.). In the repo
  this is `config.DIAGNOSES_DIR`; in Colab cells it's the `DIAG` variable. Non-video outputs
  (CSV/JSON) stay in the Drive project root or `Batting Key Point`.
- **3D skeleton renders use a front-on camera** (`elev=8, azim=-85`; axis mapping dim0=lateral,
  dim1=depth, up=−dim2) so the simulation matches how the source video is shot.

## Agent Team (Feedback Engine)

Three agents make up the feedback engine, run in a fixed sequence: **MEDA → APA → APE**. Right now these are conceptual roles used to structure research, prompts, and decisions; they get implemented as real code (LLM calls, scripts, or services) once the pipeline layers feeding into them are validated.

### 1. MEDA: Mathematical and Engineering Development for Athlete

**Role:** Defines the engineering target. Given an athlete's physical inputs (height, weight, stance, batting style: contact, level-swing, or power hitter), MEDA derives or references the biomechanical equations and reasoning for what an "ideal" swing looks like for that athlete profile. **Input:** Athlete physical profile (height, weight, stance, batting style category). **Output:** A written engineering rationale, the target mechanics and the math/physics reasoning behind them, for this specific athlete type. This is the baseline MEDA, APA, and APE all measure against. **Note:** Treat this as a strategy/engineering-reasoning task, not a live calculation, until the underlying biomechanics research is locked into MEMORY.md.

### 2. APA: Athlete Performance Analyzer

**Role:** Looks at the athlete's actual swing footage (via the pose/3D-lifting/event-detection layers upstream) and extracts body parts and joint landmarks to characterize their current style. **Input:** Pose/keypoint data and detected swing events for one athlete. **Output:** A pros-and-cons breakdown of the athlete's current swing, organized by mechanical category (e.g. stance, bat path, hip-shoulder separation), referencing the same plain-language metric definitions used elsewhere (unit + what "good" looks like for a high school athlete).

### 3. APE: Athlete Performance Enhancement

**Role:** Synthesizes MEDA's ideal-mechanics target and APA's current-swing breakdown into specific, actionable corrections. **Input:** MEDA's output (target mechanics) + APA's output (current pros/cons). **Output:** Ranked recommendations for improvement, covering stance, swing style, bat path, swing angle, contact point, rotation power, and rotation timing. Recommendations should vary by athlete. APE should explicitly reference _why_ a given correction follows from this athlete's specific MEDA target and APA gaps, not give generic advice.

**Handoff convention:** Each agent's output should be saved as a discrete artifact (not just inline chat text) so the next agent in the sequence has a clean, citable input. Suggested location once implementation starts: a `feedback-engine/` subfolder with one file per agent per athlete run.

## Editorial Rules

Follow my voice principles in 00_Resources (voice-principles.md).

- Write technical documentation in plain language. Assume a smart non-engineer reader.
- When describing metrics, always include the unit and what "good" looks like for a high school athlete.
- Avoid jargon-only explanations. Pair every technical term with a one-sentence plain-English definition on first use.

Based on the YOLO_Baseball.ipynb and our agents, please write me a repository which contains src folder (coding files), and following CLAUDE.md and MEMORY.md. Don't include batting tracking coding part and  New approach with Kalman Filter Class & Wrist Proxy. You can just include cell 1, 2, 3, 4, and every cell after Bat tracking is deferred, and moving to MotionBert. 
Once you are done, using the agents, how can I effectively apply 2D domain in the video? Once you figure it out, implement it under MotionBERT, run it, and save it under Google Drive (/content/drive/My Drive/Baseball Project/). You have full access to (https://drive.google.com/drive/folders/118YuXYC0FlBbb52C-OwcsHb-eoBHkQnI?usp=sharing) and able to write a file on your own.