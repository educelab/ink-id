# cython: language_level=3
"""
Some math utils for vector rotation using quaternions.

These are used for orienting the basis vectors before sampling subvolumes, among other things. Ported from
https://gitlab.com/ideasman42/blender-mathutils (from C to Cython) under GPLv2.

"""

cimport libc.math as math
import sys

cdef float normalize_v3_v3_length(float r[3], const float a[3], const float unit_length):
    cdef float d = dot_v3v3(a, a)

    # a larger value causes normalize errors in a scaled down models with camera extreme close
    if d > 1.0e-35:
        d = math.sqrt(d)
        mul_v3_v3fl(r, a, unit_length / d)
    else:
        zero_v3(r)
        d = 0.0
    return d

cdef float normalize_v3_v3(float r[3], const float a[3]):
    return normalize_v3_v3_length(r, a, 1.0)

cdef float normalize_v3(float n[3]):
    return normalize_v3_v3(n, n)

cdef void mul_v3_v3fl(float r[3], const float a[3], float f):
    r[0] = a[0] * f
    r[1] = a[1] * f
    r[2] = a[2] * f

cdef void zero_v3(float r[3]):
    r[0] = 0.0
    r[1] = 0.0
    r[2] = 0.0

cdef void cross_v3_v3v3(float r[3], const float a[3], const float b[3]):
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]

cdef float dot_v3v3(const float a[3], const float b[3]):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

cdef float saasin(const float fac):
    if fac <= -1.0:
        return <float>-math.M_PI / 2.0
    elif fac >= 1.0:
        return <float>math.M_PI / 2.0
    else:
        return math.asin(fac)

cdef float len_v3v3(const float a[3], const float b[3]):
    cdef float d[3]

    sub_v3_v3v3(d, b, a)
    return len_v3(d)

cdef float len_v3(const float a[3]):
    return math.sqrt(dot_v3v3(a, a))

cdef void sub_v3_v3v3(float r[3], const float a[3], const float b[3]):
    r[0] = a[0] - b[0]
    r[1] = a[1] - b[1]
    r[2] = a[2] - b[2]

cdef void negate_v3_v3(float r[3], const float a[3]):
    r[0] = -a[0]
    r[1] = -a[1]
    r[2] = -a[2]

cdef float angle_normalized_v3v3(const float v1[3], const float v2[3]):
    cdef float v2_n[3]
    # This is the same as acos(dot_v3v3(v1, v2)), but more accurate
    if dot_v3v3(v1, v2) >= 0.0:
        return 2.0 * saasin(len_v3v3(v1, v2) / 2.0)
    else:
        negate_v3_v3(v2_n, v2)
        return <float>math.M_PI - 2.0 * saasin(len_v3v3(v1, v2_n) / 2.0)

cdef void axis_angle_normalized_to_quat(float q[4], const float axis[3], const float angle):
    cdef float phi = 0.5 * angle
    cdef float si = math.sin(phi)
    cdef float co = math.cos(phi)
    q[0] = co
    mul_v3_v3fl(q + 1, axis, si)

cdef void unit_qt(float q[4]):
    q[0] = 1.0
    q[1] = q[2] = q[3] = 0.0

cdef int axis_dominant_v3_single(const float vec[3]):
    cdef float x = abs(vec[0])
    cdef float y = abs(vec[1])
    cdef float z = abs(vec[2])
    return (0 if x > z else 2) if (x > y) else (1 if y > z else 2)

# Calculates p - a perpendicular vector to v
cdef void ortho_v3_v3(float out[3], const float v[3]):
    cdef int axis = axis_dominant_v3_single(v)

    if axis == 0:
        out[0] = -v[1] - v[2]
        out[1] = v[0]
        out[2] = v[0]
    elif axis == 1:
        out[0] = v[1]
        out[1] = -v[0] - v[2]
        out[2] = v[1]
    elif axis == 2:
        out[0] = v[2]
        out[1] = v[2]
        out[2] = -v[0] - v[1]

cdef void axis_angle_to_quat(float q[4], const float axis[3], const float angle):
    cdef float nor[3]
    if normalize_v3_v3(nor, axis) != 0.0:
        axis_angle_normalized_to_quat(q, nor, angle)
    else:
        unit_qt(q)

# Note: expects vectors to be normalized
cdef void rotation_between_vecs_to_quat(float q[4], const float v1[3], const float v2[3]):
    cdef float angle
    cdef float axis[3]

    cross_v3_v3v3(axis, v1, v2)

    if normalize_v3(axis) > sys.float_info.epsilon:
        angle = angle_normalized_v3v3(v1, v2)
        axis_angle_normalized_to_quat(q, axis, angle)
    else:
        # Degenerate case
        if dot_v3v3(v1, v2) > 0.0:
            # Same vectors, zero rotation
            unit_qt(q)
        else:
            # Colinear but opposed vectors, 180 rotation
            ortho_v3_v3(axis, v1)
            axis_angle_to_quat(q, axis, <float>math.M_PI)

cdef void vector_rotation_difference(float q[4], const float vec_a_in[3], const float vec_b_in[3]):
    cdef float vec_a[3]
    cdef float vec_b[3]
    copy_v3_v3(vec_a, vec_a_in)
    copy_v3_v3(vec_b, vec_b_in)

    normalize_v3(vec_a)
    normalize_v3(vec_b)

    rotation_between_vecs_to_quat(q, vec_a, vec_b)

cdef void copy_qt_qt(float q1[4], const float q2[4]):
    q1[0] = q2[0]
    q1[1] = q2[1]
    q1[2] = q2[2]
    q1[3] = q2[3]

cdef float dot_qtqt(const float q1[4], const float q2[4]):
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

cdef void mul_qt_fl(float q[4], const float f):
    q[0] *= f
    q[1] *= f
    q[2] *= f
    q[3] *= f

cdef float normalize_qt(float q[4]):
    cdef float length = math.sqrt(dot_qtqt(q, q))
    if length != 0.0:
        mul_qt_fl(q, 1.0 / length)
    else:
        q[1] = 1.0
        q[0] = q[2] = q[3] = 0.0
    return length

cdef float normalize_qt_qt(float r[4], const float q[4]):
    copy_qt_qt(r, q)
    return normalize_qt(r)

cdef void quat_to_mat3(float m[3][3], const float q[4]):
    cdef double q0, q1, q2, q3, qda, qdb, qdc, qaa, qab, qac, qbb, qbc, qcc

    q0 = math.M_SQRT2 * <double>q[0]
    q1 = math.M_SQRT2 * <double>q[1]
    q2 = math.M_SQRT2 * <double>q[2]
    q3 = math.M_SQRT2 * <double>q[3]

    qda = q0 * q1
    qdb = q0 * q2
    qdc = q0 * q3
    qaa = q1 * q1
    qab = q1 * q2
    qac = q1 * q3
    qbb = q2 * q2
    qbc = q2 * q3
    qcc = q3 * q3

    m[0][0] = <float>(1.0 - qbb - qcc)
    m[0][1] = <float>(qdc + qab)
    m[0][2] = <float>(-qdb + qac)

    m[1][0] = <float>(-qdc + qab)
    m[1][1] = <float>(1.0 - qaa - qcc)
    m[1][2] = <float>(qda + qbc)

    m[2][0] = <float>(qdb + qac)
    m[2][1] = <float>(-qda + qbc)
    m[2][2] = <float>(1.0 - qaa - qbb)

cdef void quaternion_to_rmat(float rmat[3][3], const float quaternion[4]):
    cdef float tquat[4]
    normalize_qt_qt(tquat, quaternion)
    quat_to_mat3(rmat, tquat)

cdef void copy_v3_v3(float r[3], const float a[3]):
    r[0] = a[0]
    r[1] = a[1]
    r[2] = a[2]

cdef void mul_v3_m3v3(float r[3], const float m[3][3], const float a[3]):
    cdef float t[3]
    copy_v3_v3(t, a)

    r[0] = m[0][0] * t[0] + m[1][0] * t[1] + m[2][0] * t[2]
    r[1] = m[0][1] * t[0] + m[1][1] * t[1] + m[2][1] * t[2]
    r[2] = m[0][2] * t[0] + m[1][2] * t[1] + m[2][2] * t[2]

cdef void mul_m3_v3(const float m[3][3], float r[3]):
    cdef float a[3]
    copy_v3_v3(a, r)
    mul_v3_m3v3(r, m, a)

cdef void rotate(float out[3], const float vec[3], const float quaternion[4]):
    cdef float rmat[3][3]

    quaternion_to_rmat(rmat, quaternion)

    copy_v3_v3(out, vec)

    mul_m3_v3(rmat, out)