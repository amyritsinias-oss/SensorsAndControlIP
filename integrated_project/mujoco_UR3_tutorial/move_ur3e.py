import time
import mujoco
import mujoco.viewer
from pathlib import Path

# Path to the UR3e MJCF model
model_path = Path(__file__).parent / "universal_robots_ur3e" / "ur3e.xml"

# Load the MuJoCo model and create simulation data
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Reset to the default home keyframe defined in the MJCF
mujoco.mj_resetDataKeyframe(model, data, 0)

# Base joint actuator index ('shoulder_pan') and target velocity in rad/s
base_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan")
slow_velocity = 0.1  # rad/s (~5.7 deg/s)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Integrate velocity command into the position actuator's control target
        data.ctrl[base_actuator_id] += slow_velocity * model.opt.timestep

        # Step the physics
        mujoco.mj_step(model, data)
        viewer.sync()

        # Real-time synchronization
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
