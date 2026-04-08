import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros

def mrp_sanity_check(sigma) -> None:
    # Rotation angle
    rot_ang_rad = 4 * np.atan2(np.linalg.norm(sigma), 1)
    rot_ang_deg = rot_ang_rad*macros.R2D

    # Rotation axis
    e_hat = sigma / np.linalg.norm(sigma)

    print(f"mrp rotation angle: {rot_ang_deg}")
    print(f"rotation axis:      {e_hat}")


# XYXeul = [np.pi/4, 0, 0]
# ZYXeul = [np.pi/4, 0, 0]
# dcm = np.matrix([[  1.0000000,  0.0000000,  0.0000000],
#        [0.0000000,  0.7071068, -0.7071068],
#    [0.0000000,  0.7071068,  0.7071068 ]])


# mrp_dcm = rbk.C2MRP(dcm)
# mrp_XYXeul = rbk.euler1212MRP(XYXeul)
# mrp_ZYXeul = rbk.euler3212MRP(ZYXeul)

# # mrp_sanity_check(mrp_dcm)
# # mrp_sanity_check(mrp_XYXeul)
# # mrp_sanity_check(mrp_ZYXeul)


# r_LN_N = [1, 2, 3]
# r_BN_N = [1, 1, 1]

# r_BL_N = [a - b for a, b in zip(r_LN_N, r_BN_N)] # r_LN_N - r_BN_N

# print(r_BL_N)

# np_r_L = np.array(r_LN_N)
# np_r_B = np.array(r_BN_N)
# print(np_r_L)
# print(np_r_B)

# np_r_BL = np_r_L - np_r_B

# print(np_r_BL)


# r_BL_N_hat = np_r_BL / np.linalg.norm(np_r_BL)
# print(r_BL_N_hat)

# # Satellite face configuration
# r_PB_B = np.array([0, 0, 1])
# r_AB_B= np.array([0, 1, 0])
# # Get the available GS position relative to Earth in inertial frame
# # assert self.selectedGsIdx is not None
# # r_LN_N = np.array(self.gsStateMsgs[self.selectedGsIdx].read().r_LN_N)
# r_LN_N = np.array([100, 0, 0])
# # Get spacecraft position relative to Earth in inertial frame 
# # r_BN_N = np.array(self.nav.transOutMsg.read().r_BN_N) # TODO: Verify this
# r_BN_N = np.array([0, 0, 200])

# # =================== COMMS POINTING MODE ===================
# # Unit vector from spacecraft Body to selected ground station (desired antenna direction)
# r_LB_N = r_LN_N - r_BN_N
# r_LB_N_hat = r_LB_N / np.linalg.norm(r_LB_N)

# # Get Sun position vector relative to the Earth in inertial frame
# # r_SN_N = np.array(self.sunStateMsg.read().PositionVector)
# r_SN_N = np.array([-1000, 0, 0])

# # Unit vector from spacecraft Body to the Sun (sun vector)
# r_SB_N = r_SN_N - r_BN_N
# r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

# # Project the sun vector into the plane normal to the desired antenna direction vector)
# # https://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps/ProjectionOfVectorOntoPlane
# s = r_SB_N_hat - np.dot(r_SB_N_hat, r_LB_N_hat)/np.linalg.norm(r_LB_N_hat)**2 * r_LB_N_hat
# s_hat = s / np.linalg.norm(s)

# # TODO: Make flexible for other solar panel / antenna face configurations
# y_hat = r_LB_N_hat
# z_hat = s_hat
# x_hat = np.cross(y_hat, z_hat)
# z_hat = np.cross(x_hat, y_hat)

# print("COMMS PointingMode desired orientation:")
# print(x_hat)
# print(y_hat)
# print(z_hat)

# # Direction cosine matrix for the desired attitude
# C_ND_N = np.vstack((x_hat, y_hat, z_hat))

# print(C_ND_N)

# # Convert into desired Modified Rodrigues Parameters
# mrp_D = rbk.C2MRP(C_ND_N)

# # Publish the desired attitude
# # self.guid.sigma_R0N = mrp_D



# # =================== CHARGE POINTING MODE ===================
# # Get spacecraft position relative to Earth in inertial frame 
# r_NB_N = - r_BN_N
# r_NB_N_hat = r_NB_N / np.linalg.norm(r_NB_N)

# # Unit vector from spacecraft Body to the Sun (desired solar panel direction)
# r_SB_N = r_SN_N - r_BN_N
# r_SB_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

# # Want antenna to point nadir as much as possible
# a = r_NB_N_hat - np.dot(r_NB_N_hat, r_SB_N_hat)/np.linalg.norm(r_SB_N_hat)**2 * r_SB_N_hat
# a_hat = a / np.linalg.norm(a)

# z_hat = r_SB_N_hat
# y_hat = a_hat
# x_hat = np.cross(y_hat, z_hat)
# # z_hat = np.cross(x_hat, y_hat)

# # Direction cosine matrix for the desired attitude
# C_ND_N = np.vstack((x_hat, y_hat, z_hat))

# print("\nCHARGE PointingMode desired orientation:")
# print(C_ND_N)

# # Convert into desired Modified Rodrigues Parameters
# mrp_D = rbk.C2MRP(C_ND_N)


a: dict[str, int] = {}
a[f"test{0}"] = 100
a[f"test{1}"] = 200

for key, value in a.items():
    print(key)
    print(value)