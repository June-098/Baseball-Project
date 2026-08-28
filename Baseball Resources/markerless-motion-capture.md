[Theia3D](https://www.theiamarkerless.com/) is markerless motion capture software that uses deep learning and [[Inverse Kinematics]] to estimate 3D human pose from synchronized multi-camera video. It analyzes swing and batted-ball trajectories using two add-ons:  

- [Bat Tracking](https://www.theiamarkerless.com/bat-tracking) extends Theia3D so that the same multi-camera array captures the bat's 3D trajectory alongside the athlete's full-body kinematics, tracking the base and barrel tip frame by frame across the swing without instrumentation on the athlete or the bat.
- [Ball Tracking](https://www.theiamarkerless.com/ball-tracking) brings pitch and batted-ball trajectory into the same synchronized capture space, so body, bat, and ball can be analyzed in one coordinate system.

#### Markerless Capture for Game-Speed Mechanics

Theia3D requires no markers on the athlete, no sensors or chips on the bat, and no instrumentation on the ball. Athletes pitch and swing at full competitive intent in their everyday training clothing, using their preferred bat, which means **the captured movement is the movement that actually occurs in performance**, not the adjusted movement that happens when markers, suits, or sensors are introduced.

## Do we need external devices or wearable?
Yes, having wearable device attached to the bat measures additional layers to understand why a hitter's exit velo, attack angle, and swing path is improving or degrading. For an example, Theia3 uses Wearable athlete monitoring, which independently tracks pitches, throws, and bat swings to objectively quantify daily workloads and manage physical demands.

## How do we use the system, and how much setup, hardware, and operator time does it demand?
For programs that need minimal friction, **portable systems** are built for immediate deployment. Single-unit launch monitors require only a basic tripod and correct placement behind the plate (typically 8 to 14 feet back), which means they demand very little hardware management. Coaches can set them up in a bullpen or cage and start pulling data almost instantly without eating into practice time.

The initial setup is more involved and requires line-of-sight management to ensure the entire capture area is covered. The tradeoff is favorable for permanent or semi-permanent labs: once the initial calibration is established, the system is reusable across sessions, and coaches can continuously capture full 3D body, bat, and ball data without outfitting players in markers.

**Athlete monitoring systems** operate on a different logistical model. The hardware tracks the athlete rather than the environment, so these systems require zero stadium or cage infrastructure. The technology captures swings, pitches, and throws via a single wearable device worn by the player.

## Where does your Data live and who owns it?
Programs using **fixed multi-camera stadium installations** generally don’t handle the raw tracking data themselves. Captured video is processed by the vendor's proprietary software, transferred through cloud infrastructure, and returned to teams and broadcast partners as finalized metrics. Teams depend on the vendor's pipeline and have limited control over the raw capture environment, the processing methodology, and how long source data is retained.

**Wearables and portable launch monitors** typically push to vendor cloud dashboards. Athlete-monitoring wearables and most portable radar and camera-based launch monitors are built around proprietary cloud ecosystems.

## Where does it run and how does it run?
Theia3D's desktop application runs on consumer-grade NVIDIA GPUs and uses the parameters from the camera calibration to calculate where key points on the athlete's body, the baseball bat, and the ball are in 3D space. 

These points are then fit to a 3D skeleton based on user-specified joint constraints. The work is compute-intensive and depends on deep-learning models trained on more than 100 million images spanning more than 1,000 different environments, **allowing the system to track 124 anatomical landmarks on every visible person in every video frame**, plus the bat's base and tip on every frame the bat is visible.

Theia3D records the ball's position frame by frame, combining the viewpoints of the multi-camera array to reconstruct the continuous 3D trajectory of the ball's flight. This camera-based approach **calculates ball velocity accurately by tracking how the ball's position changes across successive video frames**. Because the camera array records the entire environment rather than solely tracking the ball, the system directly captures the exact release point from the pitcher's hand and the point of contact with the bat, alongside the athletes themselves.

**The result is a detailed, synchronized 3D swing profile** that links the bat's complete trajectory (including metrics like bat speed, bat path, and timing) directly to the player's full-body biomechanics. As for the human body itself, a 3D skeleton is generated with 17 body segments, allowing users to measure joint angles, movement patterns, and other details about how the person is moving.

Once processing is finished, users can save the 3D motion data in standard file formats like .C3D, .FBX, or .JSON, which makes it easy to use the data in software like Visual3D and other downstream programs. When exporting to .C3D files, the software saves both raw unfiltered poses and smoothed filtered poses, giving users the option to work with either version in downstream analysis.

This approach is a significant departure from bat-mounted sensors (e.g., knob sensors, embedded smart bats), which rely on accelerometers and gyroscopes and have to backtrack from those signals to derive position and angle.

