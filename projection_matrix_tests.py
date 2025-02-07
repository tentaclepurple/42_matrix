from LinearAlgebra import Matrix, Vector
from LinearAlgebra import lerp


def save_to_proj_file(matrix: Matrix, folder_path: str = "matrix_display") -> None:
    """
    Saves the matrix to a 'proj' file in the specified folder.
    Creates the folder if it doesn't exist.
    The matrix is saved in column-major order with the format required by the display software.
    
    Args:
        folder_path: Path to the folder where the proj file will be saved
    """
    import os
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Full path for the proj file
    file_path = os.path.join(folder_path, "proj")
    
    # Write matrix to file
    with open(file_path, "w") as f:
        rows, cols = matrix.shape()
        for i in range(rows):
            line = ", ".join(f"{matrix.data[i][j]:.6f}" for j in range(cols))
            f.write(line + "\n")



def test_projection():
    from math import pi
    
    # Create projection matrix
    proj_matrix = Matrix.projection(
        fov=60.0 * pi / 180.0,  # 60 degrees in radians
        ratio=1,          # Aspect ratio
        near=10.1,               # Near plane
        far=10.0               # Far plane
    )
    
    # Save to proj file
    save_to_proj_file(proj_matrix)
    print(f"Matrix:\n{proj_matrix}\nSaved to 'matrix_display/proj' file")


if __name__ == "__main__":
    try:        
        test_projection()

    except Exception as e:
        print(f"Error: {e}")
