from LinearAlgebra import Matrix, Vector
from LinearAlgebra import lerp
import time
import random



M1 = Matrix([[2, 1], [3, 4]])
M2 = Matrix([[20, 10], [30, 40]])

V1 = Vector([2 + 2j, 1 + 1j]) 
V2 = Vector([4 + 1j, 2 + 2j])

V3 = V1.dot(V2)
print(V3)
