from Ballast import Ballast

import numpy as np
from scipy.spatial.transform import Rotation as R


def adjoint(w):
    return np.array([[0,-w[2, 0], w[1, 0]] , [w[2, 0], 0, -w[0, 0]] , [-w[1, 0], w[0, 0], 0]])

def random_ballasts(n=4):
    s = np.random.uniform(0, 1, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    x = np.random.uniform(-1., 1., n)
    r = np.sqrt(s) * 0.5
    y = r * np.cos(theta)
    z = r * np.sin(theta)
    return [Ballast(np.array([[x[i]], [y[i]], [z[i]]])) for i in range(n)]


class Object:
    def __init__(self):

        # Origin of the robot
        self.O = np.array([[0], [0], [0]])

        # Mass of the robot
        self.mass = 1.5

        # Gravity
        self.g = 9.81
        
        # Center of gravity of the robot with empty ballasts
        # self.cog = np.array([[0.05], [-0.01], [-0.15]])
        self.cog = np.array([[0.], [0.], [-0.05]])

        # Inertia matrix of the robot
        self.I = np.array([[1, 0, 0], [0, 3, 0], [0, 0, 3]])

        # Volume of the robot
        self.volume = 0.002

        # Fluid density
        self.rho = 1000.

        # Center of volume of the robot
        self.cov = np.array([[0], [0], [0]])

        # Damping matrix
        self.D = 3. * np.eye(3)

        # State vector X = [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
        self.X = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0]], dtype=np.float64)

        # Ballast
        self.ballasts = random_ballasts(n=6)
        # self.ballasts = [
        #     Ballast(np.array([[1], [0], [0]])),
        #     Ballast(np.array([[0], [0.5], [0]])),
        #     Ballast(np.array([[-0.8], [np.sqrt(2)/2], [np.sqrt(2)/2]])),
        #     Ballast(np.array([[0], [0], [0]])),
        # ]

    def update(self, h):
        # p update
        self.X[:3] += h * self.R @ self.v_r
        # vr update
        self.X[3:6] += h * (self.f_r / self.mass - np.cross(self.omega_r.T, self.v_r.T).T)
        # R update
        self.X[6:10] = R.from_matrix(self.R + h * (self.R @ adjoint(self.omega_r))).as_quat().reshape((4, 1))
        # omega_r update
        self.X[10:] += h * (np.linalg.inv(self.I) @ (self.tau_r - np.cross(self.omega_r.T, (self.I @ self.omega_r).T).T))

    def control_depth(self, u):
        error = self.depth - u
        for b in self.ballasts:
            b.mass += 0.4 * error + 0.75 * self.v_r[2, 0]

    def control_metacenter(self):
        if np.linalg.norm(self.omega_r) < 0.01:
            quantity = 0.005

            # Compute lowest and highest ballast
            lowest_ballast = self.ballasts[np.argmin([(self.R.T @ b.position)[2, 0] for b in self.ballasts if (b.mass >= b.min_mass + quantity)])]
            highest_ballast = self.ballasts[np.argmax([(self.R.T @ b.position)[2, 0] for b in self.ballasts if (b.mass <= b.max_mass - quantity)])]

            if lowest_ballast.mass >= quantity and highest_ballast.mass <= highest_ballast.max_mass - quantity:
                lowest_ballast.mass -= quantity
                highest_ballast.mass += quantity

    def H(self):
        return np.block([[R.from_quat(self.X[6:10, 0]).as_matrix(), self.X[0:3]], [np.zeros((1, 3)), 1]])
    
    @property
    def p(self):
        return self.X[0:3]
    
    @property
    def R(self):
        return R.from_quat(self.X[6:10, 0]).as_matrix()
    
    @property
    def v_r(self):
        return self.X[3:6]
    
    @property
    def omega_r(self):
        return self.X[10:]
    
    @property
    def depth(self):
        return self.X[2, 0]
    
    @property
    def G(self):
        return (self.mass * self.cog + np.sum([b.mass * b.position for b in self.ballasts])) / (self.mass + np.sum([b.mass for b in self.ballasts]))
    
    @property
    def f_r(self):
        if self.depth < 0:
            return self.R.T @ np.array([[0], [0], [(self.rho * self.volume - self.mass - np.sum([b.mass for b in self.ballasts])) * self.g]]) - self.D @ self.v_r
        else:
            return self.R.T @ np.array([[0], [0], [(- self.mass - np.sum([b.mass for b in self.ballasts])) * self.g]]) - self.D @ self.v_r
    
    @property
    def tau_r(self):
        return np.cross((self.cog - self.G).T, (self.R.T @ np.array([[0], [0], [- self.mass * self.g]])).T).T \
               + np.cross((self.cov - self.G).T, (self.R.T @ np.array([[0], [0], [self.rho * self.volume * self.g]])).T).T \
               - self.D @ self.omega_r \
               + np.sum([np.cross((b.position - self.G).T, (self.R.T @ np.array([[0], [0], [-b.mass * self.g]])).T) for b in self.ballasts], axis=0).T

    def show(self, ax):
        # Show the robot
        x, y, z = self.robot_cylinder()
        ax.plot_surface(x, y, z, color="black", alpha=0.1)

        # ax axis equalization
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Plot the inertia ellipsoid
        # x, y, z = self.inertia_ellipsoid()
        # ax.plot_surface(x, y, z, color="crimson", alpha=0.1)

        # Plot the center of gravity
        X = self.H() @ np.vstack((self.G, np.ones(1)))
        ax.scatter(X[0], X[1], X[2], color="crimson")

        # Plot the center of volume
        X = self.H() @ np.row_stack((self.cov, np.ones(1)))
        ax.scatter(X[0], X[1], X[2], color="teal")

        # Plot the ballasts
        for b in self.ballasts:
            X = self.H() @ np.row_stack((b.position, np.ones(1)))
            ax.scatter(X[0], X[1], X[2], color=b.cm.to_rgba(b.mass))

        return ax
    
    def robot_cylinder(self):
        # Showing the robot
        x = np.linspace(-1, 1, 3)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, x_grid = np.meshgrid(theta, x)
        y_grid = 0.5 * np.cos(theta_grid)
        z_grid = 0.5 * np.sin(theta_grid)
        shape = x_grid.shape
        X = self.H() @ np.row_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), np.ones(x_grid.size)))
        return (X[0]).reshape(shape), (X[1]).reshape(shape), (X[2]).reshape(shape)
    
    def inertia_ellipsoid(self):
        n = 100
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)
        T = np.block([[np.eye(3), self.cog], [0, 0, 0, 1]])
        X = self.H() @ T @ np.row_stack((1/self.I[0, 0] * np.outer(np.cos(u), np.sin(v)).ravel(),
                                      1/self.I[1, 1] * np.outer(np.sin(u), np.sin(v)).ravel(),
                                      1/self.I[2, 2] * np.outer(np.ones(n), np.cos(v)).ravel(),
                                      np.ones(n**2)
                                    ))
        return X[0].reshape((n, n)), X[1].reshape((n, n)), X[2].reshape((n, n))
