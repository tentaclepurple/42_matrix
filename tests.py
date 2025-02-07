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
    print(f"Vector 1: \n{v1}")
    print(f"Vector 2: \n{v2}")

    print(f"Vector 1 + Vector 2: \n{v1.add(v2)}")
    print()


def vector_sub(size=5):

    v1 = Vector([random.randint(0, 100) for i in range(size)])
    v2 = Vector([random.randint(0, 100) for i in range(size)])
    
    print()
    print(f"Vector 1: \n{v1}")
    print(f"Vector 2: \n{v2}")
    print(f"Vector 1 - Vector 2: \n{v1.sub(v2)}")
    print()


def scalar_mult(size):
    """
    Test function that accepts different input sizes
    size: number of elements in the initial vector
    """
    # Create vector with specified size
    v = Vector([float(i) for i in range(size)])
    scalar = random.randint(0, 100)
    print(f"Vector: \n{v}")
    print(f"Scalar: \n{scalar}")
    
    v.scl(scalar)
    print(f"After scalar multiplication: \n{v}")
    print()


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
    vector = [i for i in range(0, n)]
    mat1 = Matrix.column_major([vector, vector, vector, vector, vector])
    print(f"Matrix 1:\n{mat1}")
    mat2 = Matrix.column_major([vector, vector, vector, vector, vector])
    print(f"Matrix 2:\n{mat2}")
    start = time.time()
    mat1.add(mat2)
    total_time2 = time.time() - start
    print(f"Matrix 1 + 2:\n{mat1}")
    print(type(mat1))
    print()

    print(f"Total time row major: {total_time1:.6f} seconds")
    print(f"Total time column major: {total_time2:.6f} seconds")


def matrix_sub(n=16):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    print(f"Matrix A: \n{A}")
    B = Matrix(mat)
    print(f"Matrix B: \n{B}")
    print(f"\n A - B: \n{A.sub(B)}\n")


def matrix_scalar(n=16):
    size = int(n ** 0.5)
    mat = [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]
    A = Matrix(mat)
    print(f"\nMatrix A: \n{A}")
    scalar = random.randint(0, 100)
    print(f"\nscalar: {scalar}")
    print(f"\n A * scalar: \n{A.scl(scalar)}\n")


def vector_linear_combination(n=16):
    size = int(n ** 0.5)
    start = time.time()
    vectors = [Vector([random.randint(0, 100) for _ in range(size)]) for _ in range(3)]
    print("\nVectors: \n", vectors)
    coefficients = [random.randint(0, 100) for _ in range(3)]
    print("\nCoefficients: \n", coefficients)

    result = Vector.linear_combination(vectors, coefficients)
    print("\nResult:\n", result)
    print(f"Total time: {time.time() - start:.6f} seconds")


def linear_interpolation():
    print(f"\nV1 = Vector([2, 1])\nV2 = Vector([4, 2])")
    V1 = Vector([2, 1])
    V2 = Vector([4, 2])
    r = lerp(V1, V2, 0.3)
    print(f"\nlerp(V1, V2, 0.3): \n{r}")
    print()
    M1 = Matrix([[2, 1], [3, 4]])
    M2 = Matrix([[20, 10], [30, 40]])
    print(f"\nM1 = Matrix([[2, 1], [3, 4]])\nM2 = Matrix([[20, 10], [30, 40]])")
    print()
    r = lerp(M1, M2, 0.5)
    print(f"\nlerp(M1, M2, 0.5): \n{r}")
    print()
    r = lerp(21., 42., 0.3)
    print(f"\nlerp(21., 42., 0.3): \n{r}")


def dot_product():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print()
    print(f"Dot product: {v1.dot(v2)}")


def norms():
    # Test Vector
    v = Vector([1., 2., 3.])
    print(f"Vector: {v}\n")
    print(f"1-norm: {v.norm_1()}")        # Should be 6.0
    print(f"2-norm: {v.norm_2()}")        # Should be 3.74165738
    print(f"inf-norm: {v.norm_inf()}")    # Should be 3.0

    # Test with negative numbers
    v = Vector([-1., -2.])
    print(f"\nVector: {v}\n")
    print(f"1-norm: {v.norm_1()}")        # Should be 3.0
    print(f"2-norm: {v.norm_2()}")        # Should be 2.236067977
    print(f"inf-norm: {v.norm_inf()}")    # Should be 2.0


def angle_cos():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(f"Vector 1: \n{v1}")
    print(f"Vector 2: \n{v2}")
    print(f"Cosine similarity: \n{v1.angle_cos(v2)}\n")


def cross_product():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(f"Vector 1: \n{v1}")
    print(f"Vector 2: \n"{v2})
    print(f"Cross product: \n{v1.cross_product(v2)}")


def matrix_vector_mult():
    m1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v = Vector([1, 2, 3])
    print(f"Matrix: \n{m1}")
    print(f"Vector: \n{v}")
    print(f"Matrix x Vector: \n{m1.mul_vec(v)}")
    print()
    m2 = Matrix([[1, 2], [3, 4], [5, 6]])
    m3 = Matrix([[1, 2, 3], [4, 5, 6]])
    print(f"Matrix 2: \n{m2}")
    print(f"Matrix 3: \n{m3}")
    print(f"Matrix 2 x Matrix 3:\n{m2.mul_mat(m3)}")
    

def trace():
    mat = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Matrix: \n{mat}")
    print(f"Trace: \n{mat.trace()}")


def transpose():
    mat = Matrix([[1, 2], [5, 6], [8, 9]])
    print(f"Matrix: \n{mat}")
    print(f"Transpose: \n{mat.transpose()}")


def row_echelon_form():
    mat = Matrix([[8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.]])
    print(f"Matrix: \n{mat}\n")
    print(f"Row echelon form: \n{mat.row_echelon()}\n")
    print()
    mat2 = Matrix([[1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.]])
    print(f"Matrix: \n{mat2}\n")
    print(f"Row echelon form: \n{mat2.row_echelon()}\n")



def determinant():
    mat = Matrix([
        [ 8., 5., -2., 4.],
        [ 4., 2.5, 20., 4.],
        [ 8., 5., 1., 4.],
        [28., -4., 17., 1.]])
    print(f"Matrix: \n{mat}")
    print(f"Determinant: \n{mat.determinant()}")


def inverse():
    mat = Matrix([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.]])

    print(f"Matrix->: \n{mat}")
    inv = mat.inverse()
    print(f"Inverse: \n{inv}")

    print(f"Check: \n{mat.mul_mat(inv)}")


def test_rank():

    m1 = Matrix([[1., 0., 0.], 
                 [0., 1., 0.], 
                 [0., 0., 1.]])
    print(f"Matrix: \n{m1}\n")
    print(f"Rank: {m1.rank()}\nShould be 3")
    print()

    m2 = Matrix([[ 1., 2., 0., 0.],
            [ 2., 4., 0., 0.],
            [-1., 2., 1., 1.]
            ])
    print(f"Matrix: \n{m2}\n")
    print(f"Rank: {m2.rank()}\nShould be 2")
    print()

    m3 = Matrix([[ 8., 5., -2.],
        [ 4., 7., 20.],
        [ 7., 6., 1.],
        [21., 18., 7.]])
    print(f"Matrix: \n{m3}\n")
    print(f"Rank: {m3.rank()}\nShould be 3")
    print()

    
    


if __name__ == "__main__":
    try:        
        
        input("Testing utility functions...")
        utility()
        print()
        input("Vector add...")
        vector_add()
        print()
        input("Vector sub...")
        vector_sub()
        print()
        input("Vector scalar multiplication...")
        scalar_mult(5)
        print()
        input("Matrix add...")
        matrix_add()
        print()
        input("Matrix sub...")
        matrix_sub()
        print()
        input("Matrix x scalar...")
        matrix_scalar()
        print()
        input("Linear combination...")
        vector_linear_combination()
        print()
        input("Linear interpolation...")
        linear_interpolation()
        print()
        input("Vector dot product...")
        dot_product()
        print()
        input("Testing norms...")
        norms()
        print()
        input("Testing angle cosine...")
        angle_cos()
        print()
        input("Testing cross product...")
        cross_product()
        print()
        input("Testing matrix vector multiplication...")
        matrix_vector_mult()
        print()
        input("Testing trace...")
        trace()
        print()
        input("Testing transpose...")
        transpose()
        print()
        input("Testing Row Echelon Form...")
        row_echelon_form()
        print()
        input("Testing determinant...")
        determinant()
        print()
        input("Testing inverse...")
        inverse()
        print()
        input("Testing rank...")
        test_rank()

    except Exception as e:
        print(f"Error: {e}")
