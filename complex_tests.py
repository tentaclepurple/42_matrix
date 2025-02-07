from LinearAlgebra import Matrix, Vector
from LinearAlgebra import lerp
import random
import time


def test_complex_vector_operations():
    print("\nTesting complex vector operations...")
    
    # Test vectors with complex numbers
    v1 = Vector([1+2j, 3-1j, -2+4j])
    v2 = Vector([2-1j, 0+2j, 1+1j])
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    
    # Test addition
    print(f"v1 + v2: {v1 + v2}")
    
    # Test subtraction
    print(f"v1 - v2: {v1 - v2}")
    
    # Test scalar multiplication
    scalar = 2 + 1j
    print(f"Scalar: {scalar}")
    print(f"v1 * scalar: {v1 * scalar}")

def test_complex_dot_product():
    print("\nTesting complex dot product...")
    
    v1 = Vector([1+1j, 2-1j])
    v2 = Vector([1-2j, 3+1j])
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot product: {v1.dot(v2)}")

def test_complex_matrix_operations():
    print("\nTesting complex matrix operations...")
    
    # Test matrices with complex numbers
    m1 = Matrix([
        [1+1j, 2-1j],
        [0+2j, 3+0j]
    ])
    
    m2 = Matrix([
        [2+0j, -1+1j],
        [1-1j, 2+2j]
    ])
    
    print(f"Matrix 1:\n{m1}")
    print(f"Matrix 2:\n{m2}")
    
    # Test matrix addition
    print(f"m1 + m2:\n{m1 + m2}")
    
    # Test matrix multiplication
    print(f"m1 * m2:\n{m1.mul_mat(m2)}")
    
    # Test scalar multiplication
    scalar = 2 + 1j
    print(f"m1 * {scalar}:\n{m1 * scalar}")

def test_complex_matrix_determinant():
    print("\nTesting complex matrix determinant...")
    
    m = Matrix([
        [1+1j, 2-1j],
        [0+2j, 3+0j]
    ])
    
    print(f"Matrix:\n{m}")
    print(f"Determinant: {m.determinant()}")

def test_complex_matrix_inverse():
    print("\nTesting complex matrix inverse...")
    
    m = Matrix([
        [1+1j, 2-1j],
        [0+2j, 3+0j]
    ])
    
    print(f"Matrix:\n{m}")
    print(f"Inverse:\n{m.inverse()}")

if __name__ == "__main__":
    try:
        test_complex_vector_operations()
        test_complex_dot_product()
        test_complex_matrix_operations()
        test_complex_matrix_determinant()
        test_complex_matrix_inverse()

    except Exception as e:
        print(f"Error: {e}")