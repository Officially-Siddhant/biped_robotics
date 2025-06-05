import os
import subprocess
import numpy as np
import distutils.util
import mujoco
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt

from typing import Callable, Optional, Union, List
from google.colab import files
from IPython.display import clear_output
from plot_contact_forces import ContactForcePlotter
from cost_matrices import CostMatrix

# --- Simulation Parameters ---
DURATION = 12
FRAMERATE = 60
TOTAL_ROTATION = 15
CTRL_RATE = 0.8
BALANCE_STD = 0.01
OTHER_STD = 0.08
IMPULSE_TIME = 6.0
IMPULSE_DURATION = 0.01

# --- Load Model ---
with open('mujoco/model/humanoid/humanoid.xml', 'r') as f:
    xml = f.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# --- Find Balanced Pose ---
height_offsets = np.linspace(-0.001, 0.001, 2001)
vertical_forces = []

for offset in height_offsets:
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    data.qpos[2] += offset
    mujoco.mj_inverse(model, data)
    vertical_forces.append(data.qfrc_inverse[2])

best_offset = height_offsets[np.argmin(np.abs(vertical_forces))]
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()

# --- Control Setpoint ---
actuator_moment = np.zeros((model.nu, model.nv))
mujoco.mju_sparse2dense(
    actuator_moment,
    data.actuator_moment.reshape(-1),
    data.moment_rownnz,
    data.moment_rowadr,
    data.moment_colind.reshape(-1),
)
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
ctrl0 = ctrl0.flatten()

# --- Compute LQR Gain Matrix ---
cost = CostMatrix(model, data, qpos0)
K = cost.calc_costs()

# --- Setup External Impulse ---
IMPULSE_FORCE = np.array([300, 0, 0])
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b"torso")

# --- Setup Camera and Scene ---
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 3.3

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
model.vis.map.force = 0.01

# --- Generate Control Perturbations ---
np.random.seed(1)
nsteps = int(np.ceil(DURATION / model.opt.timestep))
perturb = np.random.randn(nsteps, model.nu)

CTRL_STD = np.empty(model.nu)
for i in range(model.nu):
    joint = model.actuator(i).trnid[0]
    dof = model.joint(joint).dofadr[0]
    CTRL_STD[i] = BALANCE_STD if dof in balance_dofs else OTHER_STD

width = int(nsteps * CTRL_RATE / DURATION)
kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
kernel /= np.linalg.norm(kernel)
for i in range(model.nu):
    perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

# --- Initialize Simulation ---
mujoco.mj_resetData(model, data)
data.qpos = qpos0
renderer = mujoco.Renderer(model, width=1280, height=720)

# --- Run Simulation ---
frames = []
step = 0
dq = np.zeros(model.nv)
plotter = ContactForcePlotter(duration=DURATION, impulse_time=IMPULSE_TIME, impulse_duration=IMPULSE_DURATION)

while data.time < DURATION:
    # Apply impulse
    if IMPULSE_TIME <= data.time < IMPULSE_TIME + IMPULSE_DURATION:
        data.xfrc_applied[body_id][:3] = IMPULSE_FORCE
    else:
        data.xfrc_applied[body_id][:3] = 0

    # LQR control
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    data.ctrl = ctrl0 - K @ dx
    data.ctrl += CTRL_STD * perturb[step]
    step += 1

    mujoco.mj_step(model, data)

    # Log and plot contact forces
    frame_forces = []
    for i in range(data.ncon):
        contact = data.contact[i]
        addr = contact.efc_address
        force = data.efc_force[addr]
        frame_forces.append(force)
    plotter.update(data.time, frame_forces)

    # Render
    if len(frames) < data.time * FRAMERATE:
        camera.azimuth = azimuth(data.time)
        renderer.update_scene(data, camera, scene_option)
        pixels = renderer.render()
        frames.append(pixels)

media.show_video(frames, fps=FRAMERATE)
plotter.finalize_and_save("contact_forces.png")