# cython: language_level=3

cdef float normalize_v3_v3_length(float r[3], const float a[3], const float unit_length)

cdef float normalize_v3_v3(float r[3], const float a[3])

cdef float normalize_v3(float n[3])

cdef void mul_v3_v3fl(float r[3], const float a[3], float f)

cdef void zero_v3(float r[3])

cdef void cross_v3_v3v3(float r[3], const float a[3], const float b[3])

cdef float dot_v3v3(const float a[3], const float b[3])

cdef float saasin(const float fac)

cdef float len_v3v3(const float a[3], const float b[3])

cdef float len_v3(const float a[3])

cdef void sub_v3_v3v3(float r[3], const float a[3], const float b[3])

cdef void negate_v3_v3(float r[3], const float a[3])

cdef float angle_normalized_v3v3(const float v1[3], const float v2[3])

cdef void axis_angle_normalized_to_quat(float q[4], const float axis[3], const float angle)

cdef void unit_qt(float q[4])

cdef int axis_dominant_v3_single(const float vec[3])

cdef void ortho_v3_v3(float out[3], const float v[3])

cdef void axis_angle_to_quat(float q[4], const float axis[3], const float angle)

cdef void rotation_between_vecs_to_quat(float q[4], const float v1[3], const float v2[3])

cdef void vector_rotation_difference(float q[4], const float vec_a_in[3], const float vec_b_in[3])

cdef void copy_qt_qt(float q1[4], const float q2[4])

cdef float dot_qtqt(const float q1[4], const float q2[4])

cdef void mul_qt_fl(float q[4], const float f)

cdef float normalize_qt(float q[4])

cdef float normalize_qt_qt(float r[4], const float q[4])

cdef void quat_to_mat3(float m[3][3], const float q[4])

cdef void quaternion_to_rmat(float rmat[3][3], const float quaternion[4])

cdef void copy_v3_v3(float r[3], const float a[3])

cdef void mul_v3_m3v3(float r[3], const float m[3][3], const float a[3])

cdef void mul_m3_v3(const float m[3][3], float r[3])

cdef void rotate(float out[3], const float vec[3], const float quaternion[4])