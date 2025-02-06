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
    
    def __mul__(self, other: 'Vector') -> Union[int, float, complex]:
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

    def cross_product(self, other: 'Vector') -> 'Vector':
        """
        Computes the cross product of two 3D vectors.
        The result is a vector perpendicular to both input vectors.
        

            
        Returns:
            Vector perpendicular to both input vectors
            
        Example:
            u = [1, 0, 0], v = [0, 1, 0]
            u × v = [0, 0, 1]
        """
        if len(self.data) != 3 or len(other.data) != 3:
            raise ValueError("Cross product only defined for 3D vectors")
            
        return Vector([
            self.data[1] * other.data[2] - self.data[2] * other.data[1],  # x
            self.data[2] * other.data[0] - self.data[0] * other.data[2],  # y
            self.data[0] * other.data[1] - self.data[1] * other.data[0]   # z
        ])
            

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
    

    def mul_vec(self, vector: 'Vector') -> 'Vector':
        """
        Matrix-Vector multiplication
        For each row of matrix:
            Compute dot product with vector
        Returns vector of results
        """
        m = len(self.data)    # rows
        n = len(self.data[0]) # cols
        
        if n != len(vector.data):
            raise ValueError("Matrix columns must match vector dimension")
            
        result = [0] * m
        for i in range(m):
            # Dot product of row i with vector
            for j in range(n):
                result[i] += self.data[i][j] * vector.data[j]
                
        return Vector(result)

    def mul_mat(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix-Matrix multiplication
        For each position (i,j) in result:
            Dot product of row i (first matrix) with column j (second matrix)
        """
        m = len(self.data)      # rows of first
        n = len(self.data[0])   # cols of first
        p = len(other.data[0])  # cols of second
        
        if n != len(other.data):
            raise ValueError("Matrix dimensions must match")
            
        result = [[0 for _ in range(p)] for _ in range(m)]
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += self.data[i][k] * other.data[k][j]
                    
        return Matrix(result)
    
    def trace(self) -> float:
        """
        Computes the trace of a square matrix.
        Trace is the sum of the elements in the main diagonal (where i == j)
        
        Returns:
            float: Sum of diagonal elements
            
        Example:
            Matrix:
            [1 2 3]
            [4 5 6]
            [7 8 9]
            Trace = 1 + 5 + 9 = 15
            
        Raises:
            ValueError: If matrix is not square
        """
        if len(self.data) != len(self.data[0]):
            raise ValueError("Trace is only defined for square matrices")
            
        n = len(self.data)
        result = 0.0
        
        # Sum elements where row index equals column index
        return sum(self.data[i][i] for i in range(len(self.data)))
    
    def transpose(self) -> 'Matrix':
        """
        Creates transpose of matrix where rows become columns.
        For a matrix A(m×n), returns A^T(n×m)
        
        Example:
            [[1, 2, 3],          [[1, 4],
            [4, 5, 6]]    ->     [2, 5],
                                [3, 6]]
        Returns:
            Matrix: Transposed matrix
        """
        if not self.data:
            raise ValueError("Cannot transpose empty matrix")
            
        # Get dimensions
        rows = len(self.data)
        cols = len(self.data[0])
        
        # Create new matrix with swapped dimensions
        # Each new row i is the old column i
        return Matrix([[self.data[j][i] for j in range(rows)] for i in range(cols)])

    def row_echelon(self) -> 'Matrix':
        """
        Convert matrix to row echelon form using previously implemented functions
        Example:
        [8  5  -2]    ->    [1   x   x]
        [4  2.5 20]         [0   1   x]
        [8  5   1]          [0   0   1]
        """
        result = Matrix([row[:] for row in self.data])
        rows = len(result.data)
        cols = len(result.data[0])

        for i in range(min(rows, cols)):
            current_row = Vector(result.data[i])
            
            # Make pivot 1 by scaling
            if current_row.data[i] != 0:
                scalar = 1.0 / current_row.data[i]
                current_row = current_row.scl(scalar)
                result.data[i] = current_row.data

            # Eliminate entries below pivot
            for j in range(i + 1, rows):
                # Convert to Vector to use our methods
                row = Vector(result.data[j])
                if row.data[i] != 0:
                    factor = -row.data[i]
                    elimination_row = current_row.scl(factor)
                    row.add(elimination_row)
                    result.data[j] = row.data

        return result
    
    def determinant(self) -> float:
        """
        Computes the determinant of a square matrix.
        For a 2x2 matrix [[a, b], [c, d]], determinant is: ad - bc
        For a 3x3 matrix, using first row expansion:
        |a b c|
        |d e f| = a|e f| - b|d f| + c|d e|
        |g h i|    |h i|   |g i|   |g h|
        
        Returns:
            float: Determinant value
            
        Raises:
            ValueError: If matrix is not square
        """
        if len(self.data) != len(self.data[0]):
            raise ValueError("Determinant only defined for square matrices")
        
        n = len(self.data)
        
        # Base cases
        if n == 1:
            return self.data[0][0]
            
        if n == 2:
            return (self.data[0][0] * self.data[1][1] - 
                    self.data[0][1] * self.data[1][0])
        
        # For 3x3 and larger, use first row expansion
        det = 1.0

        for col in range(n):
            pivot_row = col
            while pivot_row < n and self.data[pivot_row][col] == 0:
                pivot_row += 1

            if pivot_row == n:
                return 0.0

            if pivot_row != col:
                self.data[col], self.data[pivot_row] = self.data[pivot_row], self.data[col]
                det *= -1  

            pivot = self.data[col][col]
            det *= pivot 

            for row in range(col + 1, n):
                if self.data[row][col] == 0:
                    continue
                factor = self.data[row][col] / pivot
                for k in range(col, n):
                    self.data[row][k] -= factor * self.data[col][k]

        return det

def lerp(u, v, t):
    return (1 - t) * u + t * v
