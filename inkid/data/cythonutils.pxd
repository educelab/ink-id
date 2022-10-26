cdef struct cD3:
    double x
    double y
    double z

cdef struct cI3:
    int x
    int y
    int z

cdef struct cD3x3:
    cD3 x
    cD3 y
    cD3 z

cdef struct cIShape:
    int z
    int y
    int x

cdef struct cDShape:
    double z
    double y
    double x
