from LinearAlgebra import Matrix, Vector


def utility():
    # Vector test
    v = Vector([1.0, 2.0, 3.0, 4.0])
    print(f"Vector: {v}")
    print(f"Size: {v.size()}")
    
    # Convert to 2x2 matrix
    m = v.to_matrix(2, 2)
    print(f"Matrix:\n{m}")
    print(f"Shape: {m.shape()}")
    print(f"Is square: {m.is_square()}")
    
    # Convert back to vector
    v2 = m.to_vector()
    print(f"Back to vector: {v2}")


def check_types():
    v = Vector([34, 0, 23])
    print(v)

    m = Matrix([[1+2j, 2+4j], [3+1j, 4+4j]])
    print(m)


if __name__ == "__main__":
    try:        
        utility()
        check_types()

    except Exception as e:
        print(f"Error: {e}")
