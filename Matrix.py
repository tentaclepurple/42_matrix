from typing import List, TypeVar, Union
from dataclasses import dataclass
from itertools import chain


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

    def sub(self, other: 'Vector') -> 'Vector':
        """Subtracts another vector from this one"""
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        self.data = list(map(lambda x, y: x - y, self.data, other.data))
        return self

    def scl(self, scalar: Union[int, float, complex]) -> 'Vector':
        """Multiplies the vector by a scalar"""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Scalar must be numeric (int, float, or complex)")
        self.data = [x * scalar for x in self.data]
        return self


@dataclass
class Matrix:
    data: List[List[T]]  # Matrix stored as list of columns

    def __post_init__(self):
        # Check types
        #if not isinstance(self.data, list):
        #    raise TypeError("Matrix data must be a list")
        #if not all(isinstance(sublist, list) for sublist in self.data):
        #    raise TypeError("Matrix data must be a list of lists")
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
    
    def sub(self, other: 'Matrix') -> 'Matrix':
        if self.shape() != other.shape():
            raise TypeError("Both matrices must have the same shape")

        self.data = [[a - b for a, b in zip(col1, col2)] 
                 for col1, col2 in zip(self.data, other.data)]
        return self

    def scl(self, scalar: Union[int, float, complex]) -> 'Matrix':
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError("Matrix elements must be numeric (int, float, or complex)")
        
        #self.data = [[a * scalar for col] for col in self.data]

        rows, cols = self.shape()
        for i in range(rows):
            for j in range(cols):
                self.data[i][j] *= scalar


        return self
        
