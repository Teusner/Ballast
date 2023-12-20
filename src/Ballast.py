import matplotlib as mpl
import numpy as np


class Ballast:
    def __init__(self, position=np.array([[0], [0], [0]]), mass=0):
        # Ballast minimum and maximum mass
        self.min_mass = 0.
        self.max_mass = .75

        # Ballast mass
        self.mass = mass

        # Ballast position
        self.position = position

        # ColorMap
        cmap = mpl.cm.winter
        norm = mpl.colors.Normalize(vmin=self.min_mass, vmax=self.max_mass)
        self.cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    @property
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self, value):
        self._mass = np.clip(value, self.min_mass, self.max_mass)