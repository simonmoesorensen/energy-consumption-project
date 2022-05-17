from numpy import sin, sign, cos

from sklearn.base import BaseEstimator, RegressorMixin


class PhysicalModel(BaseEstimator, RegressorMixin):

    def __init__(self,
                 Ci0=0.05,   # Rolling inertia coefficient
                 Cd=0.29,    # Air drag coefficient
                 rho=1.225,  # Density of air
                 g=9.81,     # Gravitational acceleration
                 Vwind=0,    # Velocity of wind
                 A=2.33,     # Cross-sectional area of car
                 mcar=1966,  # Mass of car
                 mp=80,      # Mass of passengers
                 theta=0):   # Road slope

        self.theta = theta
        self.mp = mp
        self.mcar = mcar
        self.A = A
        self.Vwind = Vwind
        self.g = g
        self.rho = rho
        self.Cd = Cd
        self.Ci0 = Ci0

    def fit(self, V_col, ux_col):
        self.V = V_col
        self.ux = ux_col

    def predict(self, X):
        if self.V is None or self.ux is None:
            raise ValueError('Please run .fit')

        V = X[self.V]
        ux = X[self.ux]

        Crr = 0.01 * (1 + V) / 100  # Rolling resistance coefficient

        Fhc = (self.mcar + self.mp) * self.g * sin(self.theta)  # Hill climbing force
        Faero = sign(V + self.Vwind) * 1 / 2 * self.rho * self.A * self.Cd * (V + self.Vwind) ** 2  # Aerodynamic drag
        Fi = self.Ci0 * (self.mcar + self.mp) * ux  # Inertial force
        Frr = sign(V) * (self.mcar + self.mp) * self.g * cos(self.theta) * Crr  # Rolling resistance force of the wheels

        Ft = Fhc + Faero + Fi + Frr

        Pt = Ft * V
        return Pt