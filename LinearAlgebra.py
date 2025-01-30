from typing import List, TypeVar
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
        """
        String representation of matrix
        Prints each row in a new line
        """
        rows, cols = self.shape()
        result = []
        for i in range(rows):
            row = [str(self.data[j][i]) for j in range(cols)]
            result.append("[" + ", ".join(row) + "]")
        return "\n".join(result)
    
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