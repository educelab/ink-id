ctypedef unsigned short uint16

cdef struct Int3:
    int x
    int y
    int z

cdef struct Float3:
    float x
    float y
    float z

cdef struct BasisVectors:
    Float3 x
    Float3 y
    Float3 z

cdef BasisVectors get_component_vectors_from_normal(Float3 n)

