from numpy import sin, sign, cos

# Constants
Ci0 = 0.05  # Rolling inertia coefficient
Cd = 0.29  # Air drag coefficient
rho = 1.225  # Density of air
g = 9.81  # Gravitational acceleration
Vwind = 0  # Velocity of wind
A = 2.33  # Cross-sectional area of car
mcar = 1966  # Mass of car
mp = 80  # Mass of passengers
theta = 0  # Road slope


def get_traction_power(V, ux):  # Velocity, x-axis acceleration
    Crr = 0.01 * (1 + V) / 100  # Rolling resistance coefficient

    Fhc = (mcar + mp) * g * sin(theta)  # Hill climbing force
    Faero = sign(V + Vwind) * 1/2 * rho * A * Cd * (V + Vwind)**2  # Aerodynamic drag
    Fi = Ci0 * (mcar + mp) * ux  # Inertial force
    Frr = sign(V) * (mcar + mp) * g * cos(theta) * Crr  # Rolling resistance force of the wheels

    Ft = Fhc + Faero + Fi + Frr

    Pt = Ft * V
    return Pt
