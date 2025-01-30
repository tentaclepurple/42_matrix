from LinearAlgebra import Matrix, Vector


def vector_utility(size):
   """
   Test function that accepts different input sizes
   size: number of elements in the initial vector
   """
   # Create vector with specified size
   v = Vector([float(i) for i in range(size)])
   
   # Calculate dimensions for the most square matrix possible
   dim = int(size ** 0.5)
   if dim * dim < size:
       dim += 1
       
   # Convert to matrix
   m = v.to_matrix(dim, dim)
   
   # Convert back to vector
   v2 = m.to_vector()
   
   return v.size()


def matrix_utility(size):
    """
    Test matrix operations with square matrices
    """
    # Create size x size matrix
    v = Vector([float(i) for i in range(size * size)])
    m = v.to_matrix(size, size)
    shape = m.shape()
    is_square = m.is_square()
    v2 = m.to_vector()
    return v2.size()


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
