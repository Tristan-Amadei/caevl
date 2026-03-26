import numpy as np


def mat_attitude(psi, theta, phi):
    cos_psi = np.cos(psi)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    
    Bml = np.array([
        [cos_psi * cos_theta, cos_psi * sin_theta * sin_phi - sin_psi * cos_phi, cos_psi * sin_theta * cos_phi + sin_psi * sin_phi],
        [sin_psi * cos_theta, sin_psi * sin_theta * sin_phi + cos_psi * cos_phi, sin_psi * sin_theta * cos_phi - cos_psi * sin_phi],
        [-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi]
    ])
    
    return Bml


def ang_attitude(Bml):
    EPS2 = 5e-16
    
    if np.abs(Bml[2, 0]) >= (1 - EPS2):
        phi = np.sign(-Bml[2, 0] * Bml[0, 1]) * np.arccos(Bml[1, 1])
        theta = np.sign(-Bml[2, 0]) * np.pi / 2
        psi = 0
    else:
        theta = -np.arcsin(Bml[2, 0])
        ctheta = np.cos(theta)
        cphi = Bml[2, 2] / ctheta
        if np.abs(cphi) > 1:
            cphi = np.sign(cphi)
            
        phi = np.sign(Bml[2, 1]) * np.arccos(cphi)
        cpsi = Bml[0, 0] / ctheta
        if np.abs(cpsi) > 1:
            cpsi = np.sign(cpsi)
            
        psi = np.sign(Bml[1, 0]) * np.arccos(cpsi)
    
    return psi, theta, phi


def transform_rot_angle(psi_convZYX_camSSA_rad, theta_convZYX_camSSA_rad, phi_convZYX_camSSA_rad):
    Bcl=mat_attitude(psi_convZYX_camSSA_rad,theta_convZYX_camSSA_rad,phi_convZYX_camSSA_rad)
    Bcg = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    Bgl = Bcl @ Bcg.T
    cap_drone_rad, tang_drone_rad, roul_drone_rad = ang_attitude(Bgl)
    return cap_drone_rad, tang_drone_rad, roul_drone_rad