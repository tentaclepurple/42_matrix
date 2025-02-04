from typing import List, TypeVar, Union
from dataclasses import dataclass
from itertools import chain
import numpy as np

T = TypeVar('T', int, float, complex)  # T can only be float or complex


@dataclass
class Vector:
    data: List[T]

    def __post_init__(self):
        # Check types
        if not all(isinstance(x, (int, float, complex)) for x in self.data):
            raise TypeError("Vector elements must be numeric (int, float, or complex)")
        first_type = type(self.data[0])
        if not all(isinstance(x, first_type) for x in self.data):
            raise TypeError("All elements must be of the same type")
    
    def size(self) -> int:
        """Returns vector's size"""
        return len(self.data)
    
    def __str__(self) -> str:
        """String representation of vector"""
        return f"[{', '.join(str(x) for x in self.data)}]"
    
    def to_matrix(self, rows: int, cols: int) -> 'Matrix':
        """
        Converts vector to matrix
        Args:
            rows: Number of rows
            cols: Number of columns
        Returns:
            Matrix in column-major order
        Raises:
            ValueError: If dimensions don't match vector size
        """
        if rows * cols != len(self.data):
            raise ValueError("Dimensions don't match vector size")
        # Create matrix in column-major order
        matrix_data = [self.data[i::cols] for i in range(cols)]
        return Matrix(matrix_data)

    def add(self, other: 'Vector') -> 'Vector':
        """Adds another vector to this one"""
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        self.data = list(map(lambda x, y: x + y, self.data, other.data))
        return self
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Adds another vector to this one"""
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        new = list(map(lambda x, y: x + y, self.data, other.data))
        return Vector(new)

    def sub(self, other: 'Vector') -> 'Vector':
        """Subtracts another vector from this one"""
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        self.data = list(map(lambda x, y: x - y, self.data, other.data))
        return self
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtracts another vector from this one"""
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        new = list(map(lambda x, y: x - y, self.data, other.data))
        return Vector(new)

    def scl(self, scalar: Union[int, float, complex]) -> 'Vector':
        """Multiplies the vector by a scalar"""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Scalar must be numeric (int, float, or complex)")
        self.data = [x * scalar for x in self.data]
        return self
    
    def __mul__(self, scalar: Union[int, float, complex]) -> 'Vector':
        """Multiplies the vector by a scalar"""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Scalar must be numeric (int, float, or complex)")
        new = [x * scalar for x in self.data]
        return Vector(new)
    
    def __rmul__(self, scalar: Union[int, float, complex]) -> 'Vector':
        return self * scalar

    @classmethod
    def linear_combination(cls, vectors: List['Vector'], coefficients: List[Union[int, float, complex]]) -> 'Vector':
        """
        Computes the linear combination of vectors using given coefficients.
        
        Args:
            vectors: Vector list
            coefficients: Coefficient list
            
        Returns:
            Vector with the linear combination
        """

        if len(vectors) != len(coefficients):
            raise ValueError("Number of vectors must match number of coefficients")
        if list(filter(lambda x: not isinstance(x, (int, float, complex)), coefficients)):
            raise TypeError("Scalar must be numeric (int, float, or complex)")
        if not all(isinstance(x, Vector) for x in vectors):
            raise TypeError("Not Vector instances")

        
        res = cls([0] * vectors[0].size())

        for vector, coeff in zip(vectors, coefficients):
            vector.scl(coeff)
            res.add(vector)
         
        return res
    
    def dot(self, other: 'Vector') -> Union[int, float, complex]:
        """
        Computes the dot product between two vectors.
        AKA: Inner product, scalar product
        """
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        if isinstance(self.data, Vector) or isinstance(other.data, Vector):
            raise TypeError("Not Vector instances")

        return sum(x * y for x, y in zip(self.data, other.data))
    
    def __matmul__(self, other: 'Vector') -> Union[int, float, complex]:
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        if isinstance(self.data, Vector) or isinstance(other.data, Vector):
            raise TypeError("Not Vector instances")
        
        return sum(x * y for x, y in zip(self.data, other.data))
    
    def norm_1(self) -> float:
        """
        Manhattan/Taxicab norm (L1 norm)
        Example: [-1, 2, -3] -> 6 (because |−1| + |2| + |−3| = 6)
        """
        result = 0.0
        # Sum the absolute value of each component
        for x in self.data:
            result += x if x >= 0 else -x
        return result
    
    def norm_2(self) -> float:
        """
        Euclidean norm (L2 norm)
        Example: [3, 4] -> 5 (because √(3² + 4²) = √25 = 5)
        Uses dot product and power function for square root
        """
        # We can use our existing dot product implementation
        # which gives us the sum of squares
        sum_squares = self.dot(self)

        return pow(sum_squares, 0.5)
    
    def norm_inf(self) -> float:
        """
        Supremum norm (L∞ norm)
        Absolute value of the component with the largest magnitude
        Example: [1, -5, 3] -> 5 (because max(|1|, |−5|, |3|) = 5)
        """
        max_abs = 0.0
        # Find the maximum absolute value
        for x in self.data:
            current_abs = x if x >= 0 else -x
            max_abs = max(max_abs, current_abs)
        return max_abs
    
    def angle_cos(self, other: 'Vector') -> float:
        """
        Computes the cosine of the angle between two vectors using the formula:
        cos(θ) = (u·v) / (∥u∥ × ∥v∥)
        
        This implementation uses previously defined methods:
        - dot product (u·v)
        - norm (∥u∥ and ∥v∥)
        
        Examples:
            - Parallel vectors: cos(θ) = 1
            - Perpendicular vectors: cos(θ) = 0
            - Opposite direction: cos(θ) = -1
        
        Args:
            other: Vector of the same dimension
            
        Returns:
            Cosine of the angle between the vectors
        """
        if isinstance(self.data, Vector) or isinstance(other.data, Vector):
            raise TypeError("Not Vector instances")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        if not any(x != 0 for x in self.data):
            raise ValueError("Vector must not be a zero vector")
        
        # First calculate the dot product
        dot_product = self.dot(other)
        
        # Calculate the product of the magnitudes
        magnitude_product = self.norm_2() * other.norm_2()
        
        # Return the cosine
        return dot_product / magnitude_product

        
      

        

@dataclass
class Matrix:
    data: List[List[T]]  # Matrix stored as list of columns

    def __post_init__(self):
        # Check types
        all_elements = list(chain.from_iterable(self.data))
        if not all(isinstance(x, (int, float, complex)) for x in all_elements):
            raise TypeError("Matrix elements must be numeric (int, float, or complex)")
        first_type = type(all_elements[0])
        if not all(isinstance(x, first_type) for x in all_elements):
            raise TypeError("All elements must be of the same type")
    
    @classmethod
    def column_major(cls, row_major: List[List[T]]):
        column_major = [list(col) for col in zip(*row_major)]
        return cls(column_major)

    def shape(self) -> tuple[int, int]:
        """
        Returns matrix dimensions
        Returns:
            Tuple of (rows, columns)
        """
        if not self.data:
            return (0, 0)
        return (len(self.data[0]), len(self.data))
    
    def is_square(self) -> bool:
        """Checks if matrix is square (same number of rows and columns)"""
        rows, cols = self.shape()
        return rows == cols
    
    def __str__(self) -> str:
        return "[" + "\n".join(str(row) for row in self.data) + "]"
    
    def to_vector(self) -> Vector:
        """
        Converts matrix to vector in column-major order
        Returns:
            Vector containing matrix elements
        """
        rows, cols = self.shape()
        vector_data = [self.data[j][i] 
                      for j in range(cols) 
                      for i in range(rows)]
        return Vector(vector_data)
    
    def add(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise TypeError("Both matrices must have the same shape")

        """ rows, cols = self.shape()
        for j in range(cols):
            for i in range(rows):
                self.data[j][i] += other.data[j][i]
        return self """

        self.data = [[a + b for a, b in zip(col1, col2)] 
                 for col1, col2 in zip(self.data, other.data)]
        return self
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise TypeError("Both matrices must have the same shape")
        self.data = [[a + b for a, b in zip(col1, col2)] 
                 for col1, col2 in zip(self.data, other.data)]
        return self
    
    def sub(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise TypeError("Both matrices must have the same shape")

        self.data = [[a - b for a, b in zip(col1, col2)] 
                 for col1, col2 in zip(self.data, other.data)]
        return self
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise TypeError("Both matrices must have the same shape")

        self.data = [[a - b for a, b in zip(col1, col2)] 
                 for col1, col2 in zip(self.data, other.data)]
        return self

    def scl(self, scalar: Union[int, float, complex]) -> 'Matrix':
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Matrix elements must be numeric (int, float, or complex)")
        
        self.data = [[a * scalar for a in col] for col in self.data]

        return self
    
    def __mul__(self, scalar: Union[int, float, complex]) -> 'Matrix':
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Matrix elements must be numeric (int, float, or complex)")
        
        self.data = [[a * scalar for a in col] for col in self.data]

        return self
    
    def __rmul__(self, scalar: Union[int, float, complex]) -> 'Matrix':
        return self * scalar


def lerp(u, v, t):
    return (1 - t) * u + t * v
