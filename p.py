import math
from typing import List


def create_zero_matrix(rows, columns):
    zero_matrix = [[0.0] * columns for _ in range(rows)]
    return zero_matrix

# easy and best - https://www.youtube.com/watch?v=EqNcqBdrNyI

# mathematical derivation and explanations
# https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
# https://ogldev.org/www/tutorial12/tutorial12.html
# http://www.songho.ca/opengl/gl_projectionmatrix.html
# https://heinleinsgame.tistory.com/11


def projection(fov, ratio, near, far):
    # by definition  projection matrix is 4x4 matrix
    projection_matrix = create_zero_matrix(4, 4)

    fov_radian = math.radians(fov)

    # Fill in the projection matrix
    projection_matrix[0][0] = 1 / (ratio * math.tan(fov_radian / 2))
    projection_matrix[1][1] = 1 / math.tan(fov_radian / 2)
    projection_matrix[2][2] = -(far + near) / (far - near)
    projection_matrix[2][3] = -(2 * far * near) / (far - near)
    projection_matrix[3][2] = -1.0

    return projection_matrix


if __name__ == "__main__":
    fov = 100.0  # Field-of-view in degrees
    ratio = 4 / 3  # Window size ratio (width / height)
    near = 1.0  # Distance of the near plane
    far = 50.0  # Distance of the far plane

    projection_matrix = projection(fov, ratio, near, far)
    matrix_str = '\n'.join(', '.join(str(element)
                           for element in row) for row in projection_matrix)
    # print(matrix_str)
    print("Projection Matrix\n", "============\n", projection_matrix, sep='')
    print("For use in the program\n", "==================", sep='')
    print(matrix_str)