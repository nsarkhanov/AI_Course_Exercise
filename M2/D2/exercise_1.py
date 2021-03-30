import numpy as np
x = np.array([3, 6])


def alpha_magnitude(alpha, x):
    z = alpha*x
    mag = np.linalg.norm(z)
    direc = z
    if alpha > 0:

        print(
            f"Direction of x is {x} and Direction of aplhax is  {z} when alpha is {alpha}")
    if alpha < 0:
        print(
            f"Direction of x is {x} and Direction of aplhax is  {z} when alpha is {alpha}")

    if (alpha > -1) & (alpha < 1):
        mag = np.linalg.norm(z)
        mag_x = np.linalg.norm(x)
        if (mag > mag_x):
            print("Increased")
        elif mag == mag_x:
            print("Stay the same ")
        else:
            print("Decreased")


alpha_magnitude(0.6, x)
alpha_magnitude(2, x)
alpha_magnitude(-0.6, x)
alpha_magnitude(-2, x)
# returns the resulting magnitude, if the direction has changed and what has happened to the vector
