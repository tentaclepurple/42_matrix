import math

def projection(fov, ratio, near, far):

    fov = fov * 3.141592653589793 / 180 # to radians
    x = ratio * (1 / math.tan(fov / 2))
    y = 1 / math.tan(fov / 2)
    z = -(far + near) / (far - near)
    w = -(2 * far * near) / (far - near)
    P = [
        [x, 0., 0., 0.],
        [0., y, 0., 0.],
        [0., 0., z, w],
        [0., 0., 1., 0.],
    ]
    return P

def print_matrix_row_major(matrix, title="Row-major format"):
    """
    Prints matrix in row-major format and creates file in that format
    """
    print(f"\n{title}:")
    print("-" * 50)
    
    # Print to console with row-major format
    for i in range(4):
        row = ", ".join(f"{matrix[i][j]:.6f}" for j in range(4))
        print(row)
    
    # Save to file in row-major format
    """ with open('proj_row', 'w') as f:
        for i in range(4):
            row = ", ".join(f"{matrix[i][j]:.6f}" for j in range(4))
            f.write(row + "\n") """

def print_matrix_column_major(matrix, title="Column-major format"):
    """
    Prints matrix in column-major format and creates file in that format
    """
    print(f"\n{title}:")
    print("-" * 50)
    
    # Print to console with column-major format
    for i in range(4):
        col = ", ".join(f"{matrix[j][i]:.6f}" for j in range(4))
        print(col)
    
    # Save to file in column-major format
    """ with open('proj', 'w') as f:
        for i in range(4):
            col = ", ".join(f"{matrix[j][i]:.6f}" for j in range(4))
            f.write(col + "\n") """

def main():
    try:
        # Get projection matrix
        P = projection(160, 1, 25, 1)
        
        # Print both formats
        print_matrix_row_major(P)
        print_matrix_column_major(P)
        
        print("\nArchivos creados:")
        print("- 'proj_row': matriz en formato row-major")
        print("- 'proj': matriz en formato column-major")
        
    except FileNotFoundError as e:
        print(f"Error de archivo: {e}")
    except ZeroDivisionError as e:
        print(f"Error de división por cero: {e}")
    except ValueError as e:
        print(f"Error de valor: {e}")

if __name__ == "__main__":
    main()