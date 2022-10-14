"""
Some math utils for vector rotation using quaternions.

These are used for orienting the basis vectors before sampling subvolumes, among other things. Ported from
https://gitlab.com/ideasman42/blender-mathutils (from C to Python) under GPLv2.

"""
import math
import sys

import numpy as np


I3 = tuple[int, int, int]
Fl3 = tuple[float, float, float]
Fl3x3 = tuple[Fl3, Fl3, Fl3]
Quat = tuple[float, float, float, float]


def dot_fl3(a: Fl3, b: Fl3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dot_quat(q1: Quat, q2: Quat) -> float:
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


def norm(a: Fl3) -> float:
    return math.sqrt(dot_fl3(a, a))


def normalize(a: Fl3) -> Fl3:
    d: float = dot_fl3(a, a)

    # a larger value causes normalize errors in a scaled down models with camera extreme close
    if d > 1.0e-35:
        d = math.sqrt(d)
        return mul_fl3(a, 1.0 / d)
    else:
        return 0.0, 0.0, 0.0


def cross(a: Fl3, b: Fl3) -> Fl3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def saasin(fac: float) -> float:
    if fac <= -1.0:
        return -math.pi / 2.0
    elif fac >= 1.0:
        return math.pi / 2.0
    else:
        return math.asin(fac)


def sub(a: Fl3, b: Fl3) -> Fl3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def distance(a: Fl3, b: Fl3) -> float:
    return norm(sub(b, a))


def negate(a: Fl3) -> Fl3:
    return -a[0], -a[1], -a[2]


def angle_normalized(a: Fl3, b: Fl3) -> float:
    # This is the same as acos(dot(a, b)), but more accurate
    if dot_fl3(a, b) >= 0.0:
        return 2.0 * saasin(distance(a, b) / 2.0)
    else:
        b = negate(b)
        return math.pi - 2.0 * saasin(distance(a, b) / 2.0)


def unit_qt() -> Quat:
    return 1.0, 0.0, 0.0, 0.0


def mul_fl3(a: Fl3, f: float) -> Fl3:
    return a[0] * f, a[1] * f, a[2] * f


def mul_quat(q: Quat, f: float) -> Quat:
    return q[0] * f, q[1] * f, q[2] * f, q[3] * f


def axis_angle_normalized_to_quat(axis: Fl3, angle: float) -> Quat:
    phi: float = 0.5 * angle
    si: float = math.sin(phi)
    co: float = math.cos(phi)
    m = mul_fl3(axis, si)
    return co, m[0], m[1], m[2]


def dominant_axis(a: Fl3) -> int:
    x: float = abs(a[0])
    y: float = abs(a[1])
    z: float = abs(a[2])
    return (0 if x > z else 2) if (x > y) else (1 if y > z else 2)


def ortho(a: Fl3) -> Fl3:
    """Returns a vector perpendicular to a"""
    axis: int = dominant_axis(a)

    if axis == 0:
        return -a[1] - a[2], a[0], a[0]
    elif axis == 1:
        return a[1], -a[0] - a[2], a[1]
    elif axis == 2:
        return a[2], a[2], -a[0] - a[1]


def axis_angle_to_quat(axis: Fl3, angle: float) -> Quat:
    if norm(axis) != 0.0:
        normalized = normalize(axis)
        return axis_angle_normalized_to_quat(normalized, angle)
    else:
        return unit_qt()


def rotation_between_vecs_to_quat(a: Fl3, b: Fl3) -> Quat:
    a = normalize(a)
    b = normalize(b)

    axis: Fl3 = cross(a, b)

    axis = normalize(axis)
    if norm(axis) > sys.float_info.epsilon:
        angle: float = angle_normalized(a, b)
        return axis_angle_normalized_to_quat(axis, angle)
    else:
        # Degenerate case
        if dot_fl3(a, b) > 0.0:
            # Same vectors, zero rotation
            return unit_qt()
        else:
            # Colinear but opposed vectors, 180 rotation
            axis = ortho(a)
            return axis_angle_to_quat(axis, math.pi)


def normalize_qt(q: Quat) -> Quat:
    length: float = math.sqrt(dot_quat(q, q))
    if length != 0.0:
        return mul_quat(q, 1.0 / length)
    else:
        return 0.0, 1.0, 0.0, 0.0


def quat_to_rmat(q: Quat) -> Fl3x3:
    q0: float = math.sqrt(2) * q[0]
    q1: float = math.sqrt(2) * q[1]
    q2: float = math.sqrt(2) * q[2]
    q3: float = math.sqrt(2) * q[3]

    qda: float = q0 * q1
    qdb: float = q0 * q2
    qdc: float = q0 * q3
    qaa: float = q1 * q1
    qab: float = q1 * q2
    qac: float = q1 * q3
    qbb: float = q2 * q2
    qbc: float = q2 * q3
    qcc: float = q3 * q3

    m0: Fl3 = (1.0 - qbb - qcc, qdc + qab, -qdb + qac)
    m1: Fl3 = (-qdc + qab, 1.0 - qaa - qcc, qda + qbc)
    m2: Fl3 = (qdb + qac, -qda + qbc, 1.0 - qaa - qbb)

    return m0, m1, m2


def quaternion_to_rmat(q: Quat) -> Fl3x3:
    nq: Quat = normalize_qt(q)
    return quat_to_rmat(nq)


def mul_fl3x3_fl3(m: Fl3x3, a: Fl3) -> Fl3:
    return (
        m[0][0] * a[0] + m[1][0] * a[1] + m[2][0] * a[2],
        m[0][1] * a[0] + m[1][1] * a[1] + m[2][1] * a[2],
        m[0][2] * a[0] + m[1][2] * a[1] + m[2][2] * a[2],
    )


def rotate(vec: Fl3, quat: Quat) -> Fl3:
    rmat: Fl3x3 = quaternion_to_rmat(quat)

    return mul_fl3x3_fl3(rmat, vec)


def get_component_vectors_from_normal(normal: Fl3) -> Fl3x3:
    """Get a subvolume oriented based on a surface normal vector.

    Calculate the rotation needed to align the z axis of the
    subvolume with the surface normal vector, and then apply that
    rotation to all three axes of the subvolume in order to get
    the vectors for the subvolume axes in the volume space.

    See:
    https://docs.blender.org/api/blender_python_api_current/mathutils.html

    """
    x_vec: Fl3 = (1.0, 0.0, 0.0)
    y_vec: Fl3 = (0.0, 1.0, 0.0)
    z_vec: Fl3 = (0.0, 0.0, 1.0)

    quat: Quat = rotation_between_vecs_to_quat(z_vec, normal)

    x_vec = rotate(x_vec, quat)
    y_vec = rotate(y_vec, quat)
    z_vec = rotate(z_vec, quat)

    return x_vec, y_vec, z_vec


def get_basis_from_square(square_corners) -> Fl3x3:
    top_left, top_right, bottom_left, bottom_right = np.array(square_corners)

    x_vec: Fl3 = tuple(((top_right - top_left) + (bottom_right - bottom_left)) / 2.0)
    y_vec: Fl3 = tuple(((bottom_left - top_left) + (bottom_right - top_right)) / 2.0)
    z_vec = cross(x_vec, y_vec)

    x_vec = normalize(x_vec)
    y_vec = normalize(y_vec)
    z_vec = normalize(z_vec)

    return x_vec, y_vec, z_vec
