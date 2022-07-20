import numpy as np
from scipy import signal
import quaternion
from scipy.spatial.transform import Rotation as R
from functools import reduce
from itertools import accumulate

# convention is wxyz quaternion

# Constants
grav_acc = 9.80665


def filter_ch(data, b, a):
    """ Applies filter (defined by 'b' and 'a') to each channel in last dim """
    if len(data.shape) == 2:
        return np.vstack([signal.filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]).T
    else:
        raise NotImplemented()


def window_data(frame_sz, stride, data):
    """Create sliding windows for data passed as numpy array"""
    indexer = np.arange(frame_sz)[
        None, :] + np.arange(0, data.shape[0] - frame_sz, stride)[:, None]
    return data[indexer]


def gyro2quat(gyro, interval=0.01):
    """ Converts gyro signal from 3 axis representation to quat representation """
    norm = np.linalg.norm(gyro*interval, axis=-1)
    return np.concatenate([np.cos(norm/2)[..., :, None], gyro*interval/norm[..., :, None] * np.sin(norm/2)[..., :, None]], axis=-1)


def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q-1] - quat[q], axis=0) > np.linalg.norm(quat[q-1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_conjugate(quat):
    """ Conjugate (inverse) of quaternion """
    conj = np.copy(quat)
    conj[..., -3:] *= -1
    return conj


def quat_multiply(quaternion0, quaternion1):
    """ Vector product of 2 quaternions quat0 * quat1 """
    w0, x0, y0, z0 = [quaternion1[..., i] for i in range(4)]
    w1, x1, y1, z1 = [quaternion0[..., i] for i in range(4)]
    return np.stack([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], -1)


def quat_disp(quat1, quat2):
    """ 
    Quaternion representing displacement from quat1 to quat2 
    quat1 * diff = quat2
    """
    return quat_multiply(quat_conjugate(quat1), quat2)


def quat_rot_dist(quat1, quat2):
    """ Quaternion distance between rotations assuming already normalized """
    diff = np.abs(np.sum(quat1 * quat2, axis=-1))
    diff = np.clip(diff, 0, 1)
    return 2 * np.arccos(diff)


def quat_rot(vect, rot):
    """ Rotate vector (as quaternion) by rotation quaternion (unit quats) """
    return quat_multiply(quat_multiply(rot, vect), quat_conjugate(rot))


def pad_euler(vect):
    """ Pads 0 to 3dim vector to allow for 4dim quat operations on it """
    return np.concatenate([np.zeros([*vect.shape[:-1], 1]), vect], -1)


def z_rotate(angle, vector):
    """ Rotates 3dim vector by an angle around the z axis using quaternions"""
    zquat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    vector = pad_euler(vector)
    return quat_rot(vector, np.repeat(zquat[None, :], vector.shape[0], 0))[..., 1:]


def z_orientation_rotate(angle, original_rotations):
    """ Rotates wxyz quaternion by an angle around the z axis """
    zquat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    return quat_multiply(zquat[None, :], original_rotations)

# ACCUMULATION CODE


def cum_quat(quat):
    """ Takes sequence of quaternions and returns their cumulative product along second to last axis """
    def mult_norm(a, b):
        prod = quat_multiply(a, b)
        return prod / np.linalg.norm(prod, axis=-1)
    return reduce(lambda a, b: mult_norm(a, b), np.split(quat, quat.shape[-2], axis=-2))


def accum_quat(quat):
    def mult_norm(a, b):
        prod = quat_multiply(a, b)
        return prod / np.linalg.norm(prod, axis=-1)
    return np.concatenate(list(accumulate(np.split(quat, quat.shape[-2], axis=-2), lambda a, b: mult_norm(a, b))), -2)


def accum_cartesian(start, pts):
    """ 
    Takes sequence of cartesian displacements and accumulates them from the start along 2nd to last axis. 
    This increases sequence length by 1. 
    """
    assert start.shape[-1] == pts.shape[-1]
    return np.add.accumulate(np.insert(pts, 0, start, axis=-2), axis=-2)


def reparam_euler2quat(angles, order='xyz'):
    euler = np.stack([np.arctan2(angles[..., 0], angles[..., 3]),
                      np.arctan2(angles[..., 1], angles[..., 4]),
                      np.arctan2(angles[..., 2], angles[..., 5])], -1)
    return euler2quat(euler, order)

# ANGLE ROTATIONS


def euler2quat(angles, order='xyz'):
    """ Takes ndarray of Euler angles (XYZ order in last dim) and converts to quaternion (WXYZ) """
    cos = np.cos(angles / 2)
    sin = np.sin(angles / 2)

    if order == 'xyz':
        q_w = cos[..., 0]*cos[..., 1]*cos[..., 2] - \
            sin[..., 0]*sin[..., 1]*sin[..., 2]
        q_x = sin[..., 0]*cos[..., 1]*cos[..., 2] + \
            cos[..., 0]*sin[..., 1]*sin[..., 2]
        q_y = cos[..., 0]*sin[..., 1]*cos[..., 2] - \
            sin[..., 0]*cos[..., 1]*sin[..., 2]
        q_z = cos[..., 0]*cos[..., 1]*sin[..., 2] + \
            sin[..., 0]*sin[..., 1]*cos[..., 2]
    elif order == 'zyx':
        q_z = cos[..., 0]*cos[..., 1]*cos[..., 2] + \
            sin[..., 0]*sin[..., 1]*sin[..., 2]
        q_y = sin[..., 0]*cos[..., 1]*cos[..., 2] - \
            cos[..., 0]*sin[..., 1]*sin[..., 2]
        q_x = sin[..., 0]*cos[..., 1]*sin[..., 2] + \
            cos[..., 0]*sin[..., 1]*cos[..., 2]
        q_w = cos[..., 0]*cos[..., 1]*sin[..., 2] - \
            sin[..., 0]*sin[..., 1]*cos[..., 2]
    return np.stack([q_w, q_x, q_y, q_z], -1)


def quat2rot(quaternions):
    """ Converts float array of wxyz quaternions to 3x3 rotation matrices """
    return quaternion.as_rotation_matrix(quaternion.from_float_array(quaternions))


def rot2euler(rotation, order='xyz'):
    """ Converts rotation matrix to euler angles in Vicon global frame (XYZ-Vertical) """
    if order == 'xyz':
        x = np.arctan2(-rotation[..., 1, 2], rotation[..., 2, 2])
        y = np.arctan2(rotation[..., 0, 2], np.sqrt(
            1 - rotation[..., 0, 2]**2))
        z = np.arctan2(-rotation[..., 0, 1], rotation[..., 0, 0])
        return np.stack([x, y, z], -1)
    elif order == 'zyx':
        z = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
        y = np.arctan2(-rotation[..., 2, 0],
                       np.sqrt(1 - rotation[..., 2, 0]**2))
        x = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
        return np.stack([z, y, x], -1)


def quat2euler(quaternions, order='xyz'):
    """ Converts quaternions to Euler angles in Vicon frame """
    return rot2euler(quat2rot(quaternions), order)


def unwrap_angle(vals):
    """
    ***DEPRECATED, use np.unwrap instead***
    Unwrap angle to remove modulo
    """
    unfold_amt = 0
    for i in range(1, len(vals)):
        vals[i-1] += unfold_amt
        if vals[i] < -2 and vals[i-1] - unfold_amt > 2:
            unfold_amt += 2*np.pi
        elif vals[i] > 2 and vals[i-1] - unfold_amt < -2:
            unfold_amt -= 2*np.pi
    vals[-1] += unfold_amt
    return vals


def shift_wrap(x):
    """ Shift from [0,2pi] to [-pi,pi] """
    return (x + np.pi) % (2 * np.pi) - np.pi


def euler_rotation(x, y, z):
    """ DEPRECATED 
        Performs XYZ euler rotation on 1D signal
    """
    def r_x(t): return np.array(
        [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

    def r_y(t): return np.array(
        [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    def r_z(t): return np.array(
        [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return r_x(x) @ r_y(y) @ r_z(z)


def series_euler_rotation(x, y, z):
    """ DEPRECATE, use numpy-quaternion or scipy rotations instead 
        Performs XYZ euler rotation on last axis of array
    """
    r_x = np.zeros((x.shape[0], 3, 3))
    r_x[:, 0, 0] = 1
    r_x[:, 1, 1] = np.cos(x)
    r_x[:, 2, 2] = np.cos(x)
    r_x[:, 1, 2] = -np.sin(x)
    r_x[:, 2, 1] = np.sin(x)

    r_y = np.zeros((x.shape[0], 3, 3))
    r_y[:, 1, 1] = 1
    r_y[:, 0, 0] = np.cos(y)
    r_y[:, 2, 2] = np.cos(y)
    r_y[:, 2, 0] = -np.sin(y)
    r_y[:, 0, 2] = np.sin(y)

    r_z = np.zeros((x.shape[0], 3, 3))
    r_z[:, 2, 2] = 1
    r_z[:, 0, 0] = np.cos(z)
    r_z[:, 1, 1] = np.cos(z)
    r_z[:, 0, 1] = -np.sin(z)
    r_z[:, 1, 0] = np.sin(z)
    return np.einsum('iab,ibc,icd->iad', r_x, r_y, r_z)
