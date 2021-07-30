# cython: language_level=3

ctypedef unsigned short uint16

cdef struct Int3:
    int x
    int y
    int z

cdef struct Float3:
    float x
    float y
    float z

cdef struct Float4:
    float a
    float b
    float c
    float d

cdef struct BasisVectors:
    Float3 x
    Float3 y
    Float3 z

cdef BasisVectors get_component_vectors_from_normal(Float3 n)

cdef BasisVectors get_basis_from_square(square_corners)
