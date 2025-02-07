import math

def projection(fov: float, ratio: float, near: float, far: float) -> list:
    """
    Creates a perspective projection matrix.
    
    Args:
        fov: Field of view in radians
        ratio: Aspect ratio (width/height)
        near: Near clipping plane distance
        far: Far clipping plane distance
    """


    # Scale factors
    x_scale = 1.0 / math.tan(fov / 2.0)
    y_scale = x_scale

    # Depth normalization
    depth_norm = (far + near) / (near - far)

    # Perspective division
    pers_div = -1.0

    # Depth mapping
    depth_map = (2.0 * far * near) / (near - far)

    # Build the matrix
    return [
        [x_scale / ratio, 0.0, 0.0, 0.0],
        [0.0, y_scale, 0.0, 0.0],
        [0.0, 0.0, depth_norm, depth_map],
        [0.0, 0.0, pers_div, 0.0]
    ]

def print_matrix(matrix: list, title: str = "Matrix") -> None:
    """
    Prints matrix in both row-major format and saves in column-major format.
    """
    # Print to console (row-major format for readability)
    print(f"\n{title} (Row-major display format):")
    print("-" * 50)
    for row in matrix:
        print(", ".join(f"{x:10.6f}" for x in row))

    # Save to file in column-major format as required by the viewer
    with open('proj', 'w') as f:
        for j in range(4):  # for each column
            column = [f"{matrix[i][j]:.6f}" for i in range(4)]
            f.write(", ".join(column) + "\n")

def main():
    try:
        # Example with 90 degrees FOV (convert to radians)
        fov = 90.0 * math.pi / 180.0
        ratio = 16.0 / 9.0
        near = 2.
        far = 50.0

        # Generate projection matrix
        proj_matrix = projection(fov, ratio, near, far)
        
        # Print matrix and save to file
        print_matrix(proj_matrix, "Projection Matrix")
        print("\nMatrix saved to 'proj' file in column-major format")
        
        # Print parameters used
        print("\nParameters used:")
        print(f"FOV: {fov * 180.0 / math.pi:.1f} degrees")
        print(f"Aspect ratio: {ratio}")
        print(f"Near plane: {near}")
        print(f"Far plane: {far}")

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()