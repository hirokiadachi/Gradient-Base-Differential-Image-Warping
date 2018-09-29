import math
import numpy as np

def load_rotation_matrix(txt_path='./../livingRoom0n.gt.sim'):
    with open(txt_path, 'r') as f:
        lines = f.read().split()

    matrix = np.asarray(lines, dtype='f').reshape(int(len(lines)/12), 3, 4)

    return matrix

def Rotation2Euler(REuler, order):
    if order is "xyz":
        euler = Rot2Fixed_zyx(REuler)
    elif order is "zyx":
        euler = Rot2Fixed_xyz(REuler)

    return euler

def Rot2Fixed_xyz(R):
    thresh = 0.00001
    if (abs(R[2, 0] - 1) > thresh and abs(R[2, 0] + 1) > thresh):
        
        theta2_0 = -math.asin(R[2, 0])
        theta2_1 = math.pi - theta2_0
        theta1_0 = math.atan2(R[2, 1] / math.cos(theta2_0), R[2, 2] / math.cos(theta2_0))
        theta1_1 = math.atan2(R[2, 1] / math.cos(theta2_1), R[2, 2] / math.cos(theta2_1))
        theta3_0 = math.atan2(R[1, 0] / math.cos(theta2_0), R[0, 0] / math.cos(theta2_0))
        theta3_1 = math.atan2(R[1, 0] / math.cos(theta2_1), R[0, 0] / math.cos(theta2_1))    
        FixedResXYZ = np.array([[theta1_0, theta2_0, theta3_0], [theta1_1, theta2_1, theta3_1]])
    else:
        theta3 = 0
        if (abs(R[2, 0] + 1) < thresh):
            theta2 = math.pi / 2
            theta1 = theta3 + math.atan2(R[0, 1], R[0, 2])
        else:
            theta2 = -math.pi / 2
            theta1 = -theta3 + math.atan2(-R[0, 1], -R[0, 2])

        FixedResXYZ = np.array([theta1, theta2, theta3])

    return FixedResXYZ

def Euler2Rot(result, order='xyz'):
    x, y, z = result[0], result[1], result[2]
    xmat = np.asarray([[math.cos(x), -math.sin(x), 0],
                       [math.sin(x), math.cos(x), 0],
                       [0, 0, 1]], dtype='f')
    ymat = np.asarray([[math.cos(y), 0, math.sin(y)],
                       [0, 1, 0],
                       [-math.sin(y), 0, math.cos(y)]], dtype='f')
    zmat = np.asarray([[1, 0, 0],
                       [0, math.cos(z), -math.sin(z)],
                       [0, math.sin(z), math.cos(z)]],dtype='f')
    if order == 'xyz':
        ROT = np.dot(np.dot(zmat, ymat), xmat)
    elif order == 'zyx':
        ROT = np.dot(np.dot(xmat, ymat), zmat)
    print(ROT)


def Rot2Fixed_zyx(R):
    thresh = 0.00001
    if (abs(R[0, 2] - 1) > thresh and abs(R[0, 2] + 1) > thresh):

        theta2_0 = math.asin(R[0, 2])
        theta2_1 = math.pi - theta2_0
        theta1_0 = math.atan2(-R[1, 2] / math.cos(theta2_0), R[2, 2] / math.cos(theta2_0))
        theta1_1 = math.atan2(-R[1, 2] / math.cos(theta2_1), R[2, 2] / math.cos(theta2_1))
        theta3_0 = math.atan2(-R[0, 1] / math.cos(theta2_0), R[0, 0] / math.cos(theta2_0))
        theta3_1 = math.atan2(-R[0, 1] / math.cos(theta2_1), R[0, 0] / math.cos(theta2_1))
        FixedResZYX = np.array([[theta1_0, theta2_0, theta3_0], [theta1_1, theta2_1, theta3_1]])
        #print(FixedResZYX)

    else:
        theta3 = 0
        if (abs(R[0, 2] - 1) < thresh):
            theta2 = math.pi / 2
            theta1 = -theta3 + math.atan2(R[2, 1], R[1, 1])
        else:
            theta2 = -math.pi / 2
            theta1 = theta3 + math.atan2(R[2, 1], R[1, 1])

        FixedResZYX = np.array([theta1, theta2, theta3])

    return FixedResZYX

if __name__ == "__main__":
    Rot_mat = load_rotation_matrix()
    REuler = Rot_mat[0]
    print(REuler)
    result = Rotation2Euler(REuler, 'xyz')
    Euler2Rot(result[0], 'xyz')
    #print(result)
    #print(result/math.pi * 180)
