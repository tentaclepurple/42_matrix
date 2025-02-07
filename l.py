import numpy as np
import subprocess

def projection(fov, ratio, near, far):
    """
    Creates a perspective projection matrix.
    Following the formula from the subject:
    [2n/(r-l)    0          (r+l)/(r-l)     0          ]
    [0           2n/(t-b)   (t+b)/(t-b)     0          ]
    [0           0          -(f+n)/(f-n)    -2fn/(f-n)  ]
    [0           0          -1               0          ]
    """
    # Calculate basic parameters
    f = 1.0 / np.tan(fov / 2.0)
    depth = far - near
    
    # Create the matrix in column-major order
    matrix = np.zeros((4, 4))
    
    # First column
    matrix[0, 0] = f / ratio
    matrix[1, 0] = 0.0
    matrix[2, 0] = 0.0
    matrix[3, 0] = 0.0
    
    # Second column
    matrix[0, 1] = 0.0
    matrix[1, 1] = f
    matrix[2, 1] = 0.0
    matrix[3, 1] = 0.0
    
    # Third column
    matrix[0, 2] = 0.0
    matrix[1, 2] = 0.0
    matrix[2, 2] = -(far + near) / depth
    matrix[3, 2] = -1.0
    
    # Fourth column
    matrix[0, 3] = 0.0
    matrix[1, 3] = 0.0
    matrix[2, 3] = -(2.0 * far * near) / depth
    matrix[3, 3] = 0.0
    
    return matrix

if __name__ == "__main__":
    # Valores que funcionaron anteriormente
    fov = np.radians(60)      # 60 grados a radianes
    ratio = 1.0               # Ratio 1:1
    near = 25.0              # Plano cercano
    far = 10.0               # Plano lejano
    
    # Generar la matriz
    proj_matrix = projection(fov, ratio, near, far)
    
    # Guardar en formato correcto
    np.savetxt('proj', proj_matrix, fmt='%.6f', delimiter=', ')
    print("Matriz generada:")
    print(proj_matrix)
    
    # Ejecutar el visualizador
    subprocess.run(["./display"])