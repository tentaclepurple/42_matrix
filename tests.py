from LinearAlgebra import Matrix, Vector
import random
import time


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
   v = m.to_vector()
   
   return v.size()



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
    print()


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


def vector_add(size=0):

    v1 = Vector([random.randint(0, 100) for i in range(10)])
    v2 = Vector([random.randint(0, 100) for i in range(10)])
    
    print()
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector 1 + Vector 2: {v1.add(v2)}")
    print(v1)
    print()


def vector_sub(size=5):

    v1 = Vector([random.randint(0, 100) for i in range(size)])
    v2 = Vector([random.randint(0, 100) for i in range(size)])
    
    print()
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector 1 - Vector 2: {v1.sub(v2)}")
    print()


def vector_add_complexity(size):
    """
    Test function that accepts different input sizes
    size: number of elements in the initial vector
    """
    # Create vector with specified size
    v1 = Vector([float(i) for i in range(size)])
    v2 = Vector([float(i) for i in range(size)])
    return v1.add(v2)


def vector_sub_complexity(size):
    """
    Test function that accepts different input sizes
    size: number of elements in the initial vector
    """
    # Create vector with specified size
    v1 = Vector([float(i) for i in range(size)])
    v2 = Vector([float(i) for i in range(size)])
    return v1.sub(v2)


def scalar_mult(size):
    """
    Test function that accepts different input sizes
    size: number of elements in the initial vector
    """
    # Create vector with specified size
    v = Vector([float(i) for i in range(size)])
    scalar = random.randint(0, 100)
    print(f"Vector: {v}")
    print(f"Scalar: {scalar}")
    
    v.scl(scalar)
    print(f"After scalar multiplication: {v}")
    print()


def scalar_mult_complexity(size):
    """
    Test function that accepts different input sizes
    size: number of elements in the initial vector
    """
    # Create vector with specified size
    v = Vector([float(i) for i in range(size)])
    scalar = random.randint(0, 100)
    v.scl(scalar)
    return v


def matrix_add():
    n = 5
    vector = [i for i in range(0, n)]
    a = Matrix([vector, vector, vector, vector, vector])
    b = Matrix([vector, vector, vector, vector, vector])
    print(f"Row major")
    print(f"Matrix A:\n{a}")
    print(f"Matrix B:\n{b}")
    start = time.time()
    a.add(b)
    total_time1 = time.time() - start

    print(f"Matrix A + B:\n{a}")
    print()
    print(f"Column major")

    mat1 = Matrix.column_major([vector, vector, vector, vector, vector])
    mat2 = Matrix.column_major([vector, vector, vector, vector, vector])
    print(f"Matrix 1:\n{mat1}")
    print(f"Matrix 2:\n{mat2}")
    start = time.time()
    mat1.add(mat2)
    total_time2 = time.time() - start
    print(f"Matrix 1 + 2:\n{mat1}")

    print(f"Total time row major: {total_time1:.6f} seconds")
    print(f"Total time column major: {total_time2:.6f} seconds")



    #print(a)


if __name__ == "__main__":
    try:        
        '''
        input("Testing utility functions...")
        utility()
        input("Vector add...")
        vector_add()

        input("Vector sub...")
        vector_sub()
        
        input("Vector scalar multiplication...")
        scalar_mult(5)
        '''

        #input("Matrix add...")
        matrix_add()



    except Exception as e:
        print(f"Error: {e}")
