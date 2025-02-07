from LinearAlgebra import Matrix, Vector
from LinearAlgebra import lerp


def save_proj_file(matrix: Matrix) -> None:
    import os

    folder_path = "matrix_display"
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
    
    fov= 45.0 * pi / 180.0
    ratio = 4 / 3
    near = 2.
    far = 20.

    print("\nParameters")
    print(f"fov: {fov * 180.0 / pi:.1f} degrees")
    print(f"ratio: {ratio}")
    print(f"near: {near}")
    print(f"far: {far}")

    # Create projection matrix
    proj_matrix = Matrix.projection(
        fov,
        ratio,
        near,
        far
    )

    # Save to proj file
    #save_proj_file(proj_matrix)
    mat = str(proj_matrix)
    print(f"\n{mat.replace('[', '').replace(']', '').replace('  ', ', ')}")


if __name__ == "__main__":
    try:        
        test_projection()

    except Exception as e:
        print(f"Error: {e}")
