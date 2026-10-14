# TB3 EKF (`sim/ekf.py`)

## What it does
- Estimates the robot's pose `[x, y, th]` and its uncertainty `P`.
- **Predict:** wheel encoders give how far each wheel turned.
- **Update:** ArUco markers at known positions give range and bearing.
- Also estimates wear on each wheel (`bL`, `bR`), so a worn wheel doesn't make the estimate drift.
- Rejects bad marker readings, and recovers if the robot gets bumped.
- The controller only needs `ekf.x` and `ekf.P`.

## Usage
```python
from sim.ekf import TB3EKF

ekf = TB3EKF(x0, P0, rp, MARKERS, cam.sigma)   # add tuning="mujoco" for MuJoCo

ekf.predict(ticks)            # encoder ticks (or ekf.predict_angles(phiL, phiR))
ekf.update(detections)        # [(marker_id, range, bearing), ...]
v, w, info = ctrl.step(ekf.x, ekf.P)
```

## Tuning
- All settings are in `EKFTuning` at the top of the file. Each one is commented.
- Main settings:
  - `k_odo`: wheel slip noise
  - `sig_b_rw`: how fast wheel wear can change
  - `r_scale`: camera noise multiplier
  - `cam_offset`: how far the camera sits ahead of the robot centre (m)
  - `gate`: outlier rejection threshold
- Presets: `default`, `mujoco`, `no_bias`, `standin_equiv`.
- Run the tuning sweep (about 3 min):
  ```bash
  python -m sim.ekf --seeds 20
  ```
- Reading the output:
  - `rms_pos`: position error. Lower is better.
  - `NEES95`: aim for about 95%. Much lower means the filter is overconfident; near 100% means it's too cautious.
  - `dock`: percentage of runs that docked within tolerance.

## Results (20 runs)
- Position error: 2.43 cm
- NEES95: 96%
- Docked: 100%

## Known limitation
- Until the camera sees a marker (about the first 5 s), the filter can only use the wheels, so errors in the starting pose aren't corrected.
- Fix: put a marker near the start, or turn on the spot before driving off.
