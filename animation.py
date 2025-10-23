import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def get_animation(states, filename="pendulum_animation.mp4"):

    # Extract angles
    theta = states[0, :]
    L = 1.0  # pendulum length

    # Compute (x, y) coordinates of the pendulum bob
    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    # --- Create the figure ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.1 * L, 1.1 * L)
    ax.set_ylim(-1.1 * L, 1.1 * L)
    ax.set_aspect('equal')
    ax.set_title("Pendulum Animation")

    # Line and point for pendulum
    (line,) = ax.plot([], [], 'o-', lw=2)

    # --- Initialization function ---
    def init():
        line.set_data([], [])
        return (line,)

    # --- Update function ---
    def update(i):
        line.set_data([0, x[i]], [0, y[i]])
        return (line,)

    # --- Create animation ---
    ani = FuncAnimation(
        fig, update, frames=len(theta), init_func=init,
        blit=True, interval=30  # interval in ms between frames
    )

    writer = FFMpegWriter(fps=30, bitrate=1800)
    ani.save(filename, writer=writer)

    print(f"Animation saved as {filename}")

    plt.show()

n = 200
t = np.linspace(0, 4*np.pi, n)
states = np.vstack((np.mod(t, 2*np.pi), np.sin(t)))  # example data

get_animation(states)