import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def get_animation(states, actions, filename="pendulum_animation.mp4"):
    theta = states[0, :]
    L = 1.0  # pendulum length

    # Compute pendulum coordinates
    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    # --- Set up figure ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.1 * L, 1.1 * L)
    ax.set_ylim(-1.3 * L, 1.1 * L)
    ax.set_aspect('equal')
    ax.set_title("Sample trajectory visualisation")

    # Pendulum line
    (line_pend,) = ax.plot([], [], 'o-', lw=2)

    # Torque text label
    torque_text = ax.text(-1.0, 1.0, '', fontsize=12, color='tab:red')

    # --- Torque arrow (stationary base, use quiver so we can update in-place) ---
    torque_base_x, torque_base_y = 0.0, -1.1 * L
    torque_scale = 1.0  # controls arrow length sensitivity

    # create a quiver with zero initial vector
    torque_arrow = ax.quiver(
        torque_base_x, torque_base_y,    # base position
        0.0, 0.0,                         # initial (dx, dy)
        angles='xy', scale_units='xy', scale=1,
        width=0.03, color='tab:red'
    )

    # --- Initialization function (ensure torque_arrow is returned) ---
    def init():
        line_pend.set_data([], [])
        torque_text.set_text('')
        # set arrow to zero-length initially
        torque_arrow.set_UVC(0.0, 0.0)
        return (line_pend, torque_text, torque_arrow)

    # --- Update function (update quiver vector in-place) ---
    def update(i):
        # Update pendulum
        line_pend.set_data([0, x[i]], [0, y[i]])

        # Update torque text
        torque_text.set_text(f"Torque = {actions[i]:.3f}")

        # Compute arrow dx according to torque (arrow points left/right only)
        arrow_dx = torque_scale * actions[i]
        torque_arrow.set_UVC(arrow_dx, 0.0)

        return (line_pend, torque_text, torque_arrow)

    # --- Create animation ---
    ani = FuncAnimation(
        fig, update, frames=len(theta),
        init_func=init, blit=False, interval=30
    )

    writer = FFMpegWriter(fps=30, bitrate=1800)
    ani.save(filename, writer=writer)

    print(f"Animation saved as {filename}")

    plt.show()
