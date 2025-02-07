import numpy as np
import subprocess

def projection(fov, ratio, near, far, offset_x=0.5, offset_y=1.0):
    """
    Creates a 4x4 perspective projection matrix with offset vanishing point.
    
    Args:
        fov: Field of view in radians
        ratio: Aspect ratio (width/height)
        near: Near plane distance
        far: Far plane distance
        offset_x: Horizontal offset from center (-1 to 1)
        offset_y: Vertical offset from center (-1 to 1)
    """
    f = 1 / np.tan(fov / 2)
    depth = far - near

    proj = np.zeros((4, 4))
    
    # Términos básicos de proyección
    proj[0, 0] = f / ratio
    proj[1, 1] = f
    proj[2, 2] = -(far + near) / depth
    proj[2, 3] = -2 * far * near / depth
    proj[3, 2] = -1
    proj[3, 3] = 0
    
    # Añadir offset al punto de fuga
    proj[0, 2] = offset_x  # Desplazamiento horizontal
    proj[1, 2] = offset_y  # Desplazamiento vertical

    return proj

if __name__ == "__main__":
    # Parámetros
    fov = np.radians(20)
    ratio = 1.
    near = 25.0
    far = 20.0
    
    # Prueba con diferentes offsets
    offset_x = 0.5  # Desplaza el punto de fuga hacia la derecha
    offset_y = -0.3  # Desplaza el punto de fuga hacia abajo

    # Generar matriz
    proj_matrix = projection(fov, ratio, near, far, offset_x, offset_y)

    print("Matriz generada:")
    print(proj_matrix)
    
    # Guardar y ejecutar
    np.savetxt('proj', proj_matrix, fmt='%.5f', delimiter=', ')
    subprocess.run(["./display"])