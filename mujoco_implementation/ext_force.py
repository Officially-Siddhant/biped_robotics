import matplotlib.pyplot as plt
import numpy as np
#%matplotlib notebook
# Parameters.
DURATION = 12         # seconds
FRAMERATE = 60        # Hz
TOTAL_ROTATION = 15   # degrees
CTRL_RATE = 0.8       # seconds
BALANCE_STD = 0.01    # actuator units
OTHER_STD = 0.08      # actuator units
IMPULSE_TIME = 3.0    #seconds
IMPULSE_DURATION = 0.01  # one step
print(model.opt.timestep)
IMPULSE_FORCE = np.array([200, 0, 0])  # example force in x-direction

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b"torso")
print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))

# Make new camera, set distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2.3

# Enable contact force visualisation.
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Set the scale of visualized contact forces to 1cm/N.
model.vis.map.force = 0.01

# Define smooth orbiting function.
def unit_smooth(normalised_time: float) -> float:
  return 1 - np.cos(normalised_time*2*np.pi)
def azimuth(time: float) -> float:
  return 100 + unit_smooth(data.time/DURATION) * TOTAL_ROTATION

# Precompute some noise.
np.random.seed(1)
nsteps = int(np.ceil(DURATION/model.opt.timestep))
perturb = np.random.randn(nsteps, nu)

# Scaling vector with different STD for "balance" and "other"
CTRL_STD = np.empty(nu)
for i in range(nu):
  joint = model.actuator(i).trnid[0]
  dof = model.joint(joint).dofadr[0]
  CTRL_STD[i] = BALANCE_STD if dof in balance_dofs else OTHER_STD

# Smooth the noise. Create a kernel that smoothes the perturbation matrix. Convolution. 
width = int(nsteps * CTRL_RATE/DURATION)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
for i in range(nu):
  perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

# Reset data, set initial pose.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
print("\n This is the sys velocity: ",data.qvel)

# New renderer instance with higher resolution.
renderer = mujoco.Renderer(model, width=1280, height=720)


# --- Setup plotting ---
plt.ion()
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f"Contact {i}")[0] for i in range(4)]
ax.set_xlim(0, DURATION)
ax.set_ylim(0, 100)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Contact Force [N]")
ax.set_title("Real-time Contact Forces")

time_vals = []
force_vals = [[] for _ in range(4)]

# --- Main sim loop ---
frames = []
step = 0
while data.time < DURATION:

    # Inject external impulse at a specific time
    if IMPULSE_TIME <= data.time < IMPULSE_TIME + IMPULSE_DURATION:
      data.xfrc_applied[body_id][:3] = IMPULSE_FORCE
    else:
      data.xfrc_applied[body_id][:3] = 0  # Reset force


    # LQR feedback
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    data.ctrl = ctrl0 - K @ dx
    data.ctrl += CTRL_STD * perturb[step]
    step += 1

    # Sim step
    mujoco.mj_step(model, data)

    # Contact forces
    frame_forces = []
    for i in range(data.ncon):
        contact = data.contact[i]
        addr = contact.efc_address
        force = data.efc_force[addr]
        frame_forces.append(force)
    while len(frame_forces) < 4:
        frame_forces.append(0.0)

    # Store + plot
    time_vals.append(data.time)
    for i in range(4):
        force_vals[i].append(frame_forces[i])
        lines[i].set_data(time_vals, force_vals[i])
    ax.set_xlim(max(0, data.time - 5), data.time + 0.1)
    plt.pause(0.001)

    # Render
    if len(frames) < data.time * FRAMERATE:
        camera.azimuth = azimuth(data.time)
        renderer.update_scene(data, camera, scene_option)
        pixels = renderer.render()
        frames.append(pixels)
media.show_video(frames, fps=FRAMERATE)
#plt.ioff()
#plt.legend(loc='upper right')
#plt.show()
  
#fig.savefig("contact_forces.png", dpi=300, bbox_inches="tight")
