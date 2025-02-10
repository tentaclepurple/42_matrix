from LinearAlgebra import Matrix, Vector
from LinearAlgebra import lerp
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


def matrix_add_complexity(n):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    a = Matrix(mat)
    b = Matrix(mat)
    a.add(b)


def matrix_scalar_complexity(n):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    scalar = 42
    A.scl(scalar)


def vector_linear_combination_complexity(n=16):
    size = int(n ** 0.5)

    vectors = [Vector([random.randint(0, 10) for _ in range(size)]) for _ in range(3)]
    coefficients = [random.randint(0, 10) for _ in range(3)]
    Vector.linear_combination(vectors, coefficients)


def linear_interpolation_vector_complexity(n=16):
    size = int(n ** 0.5)
    v1 = Vector([random.randint(0, 100) for i in range(size)])
    v2 = Vector([random.randint(0, 100) for i in range(size)])
    t = random.random()
    lerp(v1, v2, t)


def dot_product_complexity(size):
    #size = int(n ** 0.5)
    v1 = Vector([random.randint(0, 100) for i in range(size)])
    v2 = Vector([random.randint(0, 100) for i in range(size)])
    v1.dot(v2)


def norms_complexity(size):
    # Test Vector
    v = Vector([random.randint(0, 100) for i in range(size)])
    v.norm_1()
    v.norm_2()
    v.norm_inf()


def angle_cos_complexity(size):
    v1 = Vector([random.randint(0, 100) for i in range(size)])
    v2 = Vector([random.randint(0, 100) for i in range(size)])
    v1.angle_cos(v2)


def matrix_vector_mult_complexity(n):
    size = int(n ** 0.5)
    mat1 = Matrix([[random.randint(0, 100) for _ in range(size)] for _ in range(size)])
    mat2 = Matrix([[random.randint(0, 100) for _ in range(size)] for _ in range(size)])
    mat1.mul_mat(mat2)


def transpose_complexity(n):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    A.transpose()


def row_echelon_complexity(n):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    A.row_echelon()


def determinant_complexity(n):
    if n == 4:
        size = 1
    elif n == 16:
        size = 2
    elif n == 64:
        size = 4
    elif n == 256:
        size = 16
    elif n == 1024:
        size = 64
    elif n == 4096:
        size = 256
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    A.determinant()


def inverse_complexity(n):
    if n == 4:
        size = 1
    elif n == 16:
        size = 2
    elif n == 64:
        size = 4
    elif n == 256:
        size = 16
    elif n == 1024:
        size = 64
    elif n == 4096:
        size = 256
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    inv = A.inverse()


def rank_complexity(n):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    A.rank()


if __name__ == "__main__":
    try:        

        input("Vector add...")
        vector_add()

        input("Vector sub...")
        vector_sub()
        
        input("Vector scalar multiplication...")
        scalar_mult(5)

        input("Matrix add...")
        matrix_add()

        input("Matrix sub...")
        matrix_sub()

        input("Matrix x scalar...")
        matrix_scalar()

        input("Linear combination...")
        vector_linear_combination()

        input("Linear interpolation...")
        linear_interpolation()

        input("Vector dot product...")
        dot_product()

        input("Vector norms...")
        norms()

        input("Vector angle cosine...")
        angle_cos()

        input("Matrix x vector...")
        matrix_vector_mult()

        input("Matrix transpose...")
        transpose()

        input("Matrix row echelon...")
        row_echelon()
        


    except Exception as e:
        print(f"Error: {e}")
