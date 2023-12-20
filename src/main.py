from Object import Object

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import pathlib


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="main.py", description="Buoyancy simulation using ballasts")
    parser.add_argument("--fps", help="Number of fps in animation", type=int, default=25)
    parser.add_argument("--show", help="Show the animation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--imgs-path", help="Folder in which images will be generated", type=pathlib.Path)
    parser.add_argument("--plots-path", help="Folder in which images will be generated", type=pathlib.Path)
    parser.add_argument("--plot-angles", help="Plot angles at in the plot-path", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-depth", help="Plot depth at in the plot-path", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-vertical-velocity", help="Plot vertical velocity at in the plot-path", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-ballasts", help="Plot ballasts at in the plot-path", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-metacenter", help="Plot metacenter at in the plot-path", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-wrench", help="Plot metacenter at in the plot-path", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.plot_angles and args.plots_path is None:
        parser.error("--plot-angles requires --plots-path.")

    o = Object()

    t0 = 0
    tf = 120
    h = 1 / args.fps
    T = np.arange(t0, tf, h)

    history = []
    wrench_history = []
    ballasts_history = []
    G_history = []

    if args.show or args.imgs_path is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for t in T:
        # Show the animation for simulation or images
        if args.show or args.imgs_path is not None:
            ax.clear()
            ax = o.show(ax)
            ax.set_title(f"t = {t:.2f} s")

        # Pause for showing the animation
        if args.show:
            plt.pause(h)

        # Compute the next state
        history.append(o.X.copy())
        wrench_history.append(np.vstack((o.f_r, o.tau_r)))
        ballasts_history.append([b.mass for b in o.ballasts])
        G_history.append(o.G.copy())
        o.update(h)
        o.control_depth(-1.)
        o.control_metacenter()

        # Images save
        if args.imgs_path is not None:
            plt.tight_layout()
            plt.savefig(f"img/{int(t/h):0{len(str(int(tf/h)))}d}.png")

    if args.plot_angles:
        fig_angles, ax_angles = plt.subplots()
        # Convert quaternion to euler
        euler = np.array([R.from_quat(x[6:10, 0]).as_euler('xyz') for x in history])
        ax_angles.plot(T, euler[:, 0], label="Roll")
        ax_angles.plot(T, euler[:, 1], label="Pitch")
        ax_angles.plot(T, euler[:, 2], label="Yaw")
        ax_angles.legend()
        ax_angles.set_xlabel(r"Time [$s$]")
        ax_angles.set_ylabel(r"Angle [$rad$]")
        ax_angles.grid()
        ax_angles.set_ylim(-np.pi, np.pi)
        ax_angles.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax_angles.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax_angles.set_xlim(0, tf)
        plt.tight_layout()
        plt.savefig(args.plots_path / "angles.png")

    if args.plot_depth:
        fig_depth, ax_depth = plt.subplots()
        ax_depth.plot(T, np.array(history)[:, 2], label="Depth")
        ax_depth.legend()
        ax_depth.set_xlabel(r"Time [$s$]")
        ax_depth.set_ylabel(r"Depth [$m$]")
        ax_depth.grid()
        plt.savefig(args.plots_path / "depth.png")

    if args.plot_vertical_velocity:
        fig_vertical_velocity, ax_vertical_velocity = plt.subplots()
        ax_vertical_velocity.plot(T, np.array(history)[:, 5], label="Vertical velocity")
        ax_vertical_velocity.legend()
        ax_vertical_velocity.set_xlabel(r"Time [$s$]")
        ax_vertical_velocity.set_ylabel(r"Vertical velocity [$m.s^{-1}$]")
        ax_vertical_velocity.grid()
        plt.savefig(args.plots_path / "vertical_velocity.png")

    if args.plot_ballasts:
        fig_ballasts, ax_ballasts = plt.subplots()
        for i, b in enumerate(o.ballasts):
            ax_ballasts.plot(T, np.array(ballasts_history)[:, i], label=f"Ballast {i+1}")
        ax_ballasts.legend()
        ax_ballasts.set_xlabel(r"Time [$s$]")
        ax_ballasts.set_ylabel(r"Mass [$kg$]")
        ax_ballasts.grid()
        plt.savefig(args.plots_path / "ballasts.png")

    if args.plot_metacenter:
        fig_metacenter, ax_metacenter = plt.subplots()
        ax_metacenter.plot(T, np.array(G_history)[:, 0], label=r"x", color="red")
        ax_metacenter.plot(T, np.array(G_history)[:, 1], label=r"y", color="green")
        ax_metacenter.plot(T, np.array(G_history)[:, 2], label=r"z", color="blue")
        ax_metacenter.legend()
        ax_metacenter.set_xlabel(r"Time [$s$]")
        ax_metacenter.set_ylabel(r"Distance [$m$]")
        ax_metacenter.grid()
        plt.savefig(args.plots_path / "metacenter.png")

    if args.plot_wrench:
        fig_force, ax_force = plt.subplots()
        ax_force.plot(T, np.array(wrench_history)[:, 0], label=r"$F_x$", color="red")
        ax_force.plot(T, np.array(wrench_history)[:, 1], label=r"$F_y$", color="green")
        ax_force.plot(T, np.array(wrench_history)[:, 2], label=r"$F_z$", color="blue")
        ax_force.legend()
        ax_force.set_xlabel(r"Time [$s$]")
        ax_force.set_ylabel(r"Forces [$N$]")
        ax_force.grid()
        plt.savefig(args.plots_path / "forces.png")

        fig_torques, ax_torques = plt.subplots()
        ax_torques.plot(T, np.array(wrench_history)[:, 3], label=r"$\Gamma_x$", color="red")
        ax_torques.plot(T, np.array(wrench_history)[:, 4], label=r"$\Gamma_y$", color="green")
        ax_torques.plot(T, np.array(wrench_history)[:, 5], label=r"$\Gamma_z$", color="blue")
        ax_torques.legend()
        ax_torques.set_xlabel(r"Time [$s$]")
        ax_torques.set_ylabel(r"Torques [$N.m$]")
        ax_torques.grid()
        plt.savefig(args.plots_path / "torques.png")
