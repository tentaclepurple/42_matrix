import numpy as np
import subprocess

def projection(fov, ratio, near, far):
    """
    Creates a 4x4 perspective projection matrix.
    """
    f = 1 / np.tan(fov / 2)
    depth = near - far  # Invertimos la diferencia

    proj = np.zeros((4, 4))
    proj[0, 0] = f / ratio
    proj[1, 1] = f
    proj[2, 2] = (far + near) / depth  # Cambiamos el signo
    proj[2, 3] = 2 * far * near / depth  # Cambiamos el signo
    proj[3, 2] = -1
    proj[3, 3] = 0

    return proj

if __name__ == "__main__":
    fov = np.radians(60)
    ratio = 1.0
    near = 25.0    # Plano cercano más cercano
    far = 15.0   # Plano lejano ajustado

    # Generate the projection matrix
    proj_matrix = projection(fov, ratio, near, far)

    # Mostrar la matriz para verificación
    print("Matriz generada:")
    print(proj_matrix)
    
    # Save and run
    np.savetxt('proj', proj_matrix, fmt='%.5f', delimiter=', ')
    subprocess.run(["./display"])