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


def main():
    try:
        P = projection(90, 16/9, 1, 10) # fov, ratio, near, far
        # P = projection(160, 16/9, 1, 10) # fov, ratio, near, far
        # P = projection(90, 16/9, 1, 1.5) # fov, ratio, near, far
        # P = projection(90, 16/9, 10, 100) # fov, ratio, near, far


    except FileNotFoundError as e:
        print(e)
    except ZeroDivisionError as e:
        print(e)
    except ValueError as e:
        print(e)

if (__name__ == "__main__"):
    main()