import numpy as np
import scipy.constants as con
import time
import matplotlib.pyplot as plt

# This file answers the first task on Project 2: Solving Quantum systems numerically


# Following functions are for validation  only
def valid1(x):

    """

    :param x: Value for the place holder x
    :return: returns the evaluation of x using the formula in this function

    This function has an analytical solution of 3250 when evaluated between a = 1 and b = 6 and it can be used to
    verify and validate the correct implementation of the trapezoidal and simpson methods in 1D
    """
    return 12*x**3 - 9*x**2 + 2


def valid2(x):

    """
    :param x: Value for the place holder x
    :return: returns the evaluation of x using the formula in this function

    This function has an analytical solution of 1 when evaluated between a = 0 and b = pi/2 and it can be used to
    verify and validate the correct implementation of the trapezoidal and simpson methods in 1D
    """
    return np.cos(x)


def valid3(x, y, z):
    """

    :param x: The x-coordinate to evaluate the function
    :param y:  The y-coordinate to evaluate the function
    :param z: The z-coordinate to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

    This function has an analytical solution of 1 when evaluated between a = [0,0,0] and b = [1,1,1] and it can be
    used to verify and validate the correct implementation of the trapezoidal and simpson methods in 3D.
    """

    return x**2 + y**2 + z**2


def valid4(x, y, z):
    """

    :param x: The x-coordinate to evaluate the function
    :param y:  The y-coordinate to evaluate the function
    :param z: The z-coordinate to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

    This function has an analytical solution of -18 when evaluated between a = [0,-1,-3] and b = [1,2,1] and it can be
    used to verify and validate the correct implementation of the trapezoidal and simpson methods in 3D.
    """

    return 6*x*y*z


# Following are functions which needs to be integrated to obtain the respective probabilities, as per the tasks
def simple_harmonic(x):

    """
    :param x: Takes in values of x supplied from the integration function
    :return: Returns the evaluation of the function at x

    This function calculates the ground state of a 1D harmonic oscillator squared, and is then integrated to
    calculate the probability. This function has been changed to work in natural units
    """

    return np.sqrt(1 / np.pi) * np.exp(-x ** 2)


def ground_state3d(x, y, z):

    """
    :param x: The x-coordinate to evaluate the function
    :param y:  The y-coordinate to evaluate the function
    :param z: The z-coordinate to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

       This function calculates the ground state of a 1D harmonic oscillator squared, and is then integrated to
       calculate the probability
    """

    c = np.sqrt(1 / np.pi)
    return c*c*c * np.exp(-x ** 2)*np.exp(-y ** 2)*np.exp(-z ** 2)


def first_state3d(x, y, z):
    """
    :param x: The x-coordinate to evaluate the function
    :param y:  The y-coordinate to evaluate the function
    :param z: The z-coordinate to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

    This function calculates the first state of a 1D harmonic oscillator squared, and is then integrated to
    calculate the probability
       """
    return (1/np.pi)**(3/2)*(x**2 + y**2)*np.exp(-(x**2 + y**2 + z**2))


# Following are the 1D Newton Coates Methods
def trapezoidal_method(a, b, e, integrand):

    """
    :param a: lower limit of the integral
    :param b: upper limit of the integral
    :param e: the user-specified relative accuracy
    :param integrand: A pointer which should have the name of the integrand function to use
    :return i2: Returns the value of the integral using the trapezoidal method

    This method uses the extended trapezoidal method of integration to evaluate the supplied integrand between the
    limits a and b
    """
    if b <= a:
        raise ValueError("b must be greater than a to evaluate integral")
    h = b-a                         # first step size
    i1 = 0.5 * h * (integrand(a) + integrand(b))      # First trapezoid area
    mid = integrand(0.5*(a+b))                   # mid point of the first and last value
    h = h / 2
    i2 = h * (0.5 * integrand(a) + 0.5 * integrand(b) + mid)  # area of n=2 iteration of trapezoid ( two parts)
    c = 2                                   # counter value, which takes into account the two iterations just done

    while abs((i2 - i1)/ i1) > e:          # Keep iterating until relative error is below user specified
        i1 = i2                                            # Let the previous removed once be equal to previous integral
        arr = np.arange(1, 2**(c-1) + 1, 1)                 # create an array of values which will be evaluated
        l = integrand(a + (h * ((2*arr - 1)/2)))            # evaluate each value in the array
        k = np.sum(l)                                       # sum all values
        i2 = (1/2) * i1 + (1/2) * h * k                     # Update value of the integral
        h = h/2                                             # Half the step size
        c += 1                                              # Increase counter by 1
    samples = 2**(c-1) + 1
    return i2,  samples


def simpson_method(a, b, e, integrand):

    """
    :param a: lower limit of the integral
    :param b: upper limit of the integral
    :param e: the user-specified relative accuracy
    :param integrand: A pointer which should have the name of the integrand function to use
    :return s: Returns the value of the integral using the simpson method

    This method uses the extended simpson method of integration to evaluate the supplied integrand between the
    limits a and b
    """
    if b <= a:
        raise ValueError("b must be greater than a to evaluate integral")
    h = b - a  # first step size
    i = []
    s_l = []
    i1 = 0.5 * h * (integrand(a) + integrand(b))  # First trapezoid area
    i.append(i1)
    mid = integrand(0.5 * (a + b))  # mid point of the first and last value
    h = h / 2
    i2 = h * (0.5 * integrand(a) + 0.5 * integrand(b) + mid)  # area of n=2 iteration of trapezoid ( two parts)
    i.append(i2)
    c = 2  # counter value, which takes into account the two iterations just done
    s = (4/3)*i2 - (1/3) * i1
    s_l.append(i2)
    s_l.append(s)
    while abs((s_l[1] - s_l[0]) / s_l[0]) > e:  # Keep iterating until relative error is below user specified
        i[0] = i[1]
        s_l[0] = s_l[1]
        arr = np.arange(1, 2 ** (c - 1) + 1, 1)  # create an array of values which will be evaluated
        l = integrand(a + (h * ((2 * arr - 1) / 2)))  # evaluate each value in the array
        k = np.sum(l)  # Sum all values in k
        i[1] = (1 / 2) * i[0] + (1 / 2) * h * k  # Update value of the integral at position 1 in the list
        s1 = (4/3)*i[1] - (1/3) * i[0]            # calculate the simpson value using the two integrals
        s_l[1] = s1                               # update the value
        h = h / 2  # Half the step size
        c += 1  # Increase counter by 1
    samples = 2 **(c - 1) + 1
    return s_l[1], samples


# Following are the 3D Newton Coates Methods
def trapezoidal_method3d(a, b, e, integrand):

    """
    :param a: lower limit of the integrals, an array of x,y,z
    :param b: upper limit of the integral, an array of x,y,z
    :param e: the user-specified relative accuracy
    :param integrand: A pointer which should have the name of the integrand function to use
    :return i2: Returns the value of the integral using the 3d trapezoidal method
    :return c : Number of iterations required to achieve the relative accuracy
    This method uses the extended trapezoidal method of integration to evaluate a 3d supplied integrand
    between the limits a and b, which
    """
    if (b <= a).any():
        raise ValueError("b must be greater than a to evaluate integral")

    if np.size(b) != 3:
        raise ValueError("b must have the dimensions 1 x 3 as an array")
    if np.size(a) != 3:
        raise ValueError("a must have the dimensions 1 x 3 as an array")
    h = b-a                                                    # Dimension lengths
    hi = np.average(h)
    i1 = 0.5 * hi * (integrand(a[0], a[1], a[2]) + integrand(b[0], b[1], b[2]))    # First trapezoid volume using x,y,z
    mid = 0.5*(a+b)
    middle = integrand(mid[0], mid[1], mid[2])       # mid point of the first and last value in each coordinate system
    hi = hi / 2
    i2 = hi*(0.5 * integrand(a[0], a[1], a[2]) + 0.5 * integrand(b[0], b[1], b[2]) + middle) # trapezoid volume, n = 2
    c = 2                                   # counter value, which takes into account the two iterations just done
    h = h/2
    while abs((i2 - i1) / i1) > e:          # Keep iterating until relative error is below user specified
        i1 = i2                             # Let the previous removed once be equal to previous integral
        k = []                              # storage of the summation in the formula
        for i in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the x domain
            for j in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the y domain
                for m in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the z domain
                    xi = a[0] + (h[0] * ((2*i - 1)/2))          # new value of x
                    yj = a[1] + (h[1] * ((2 * j - 1) / 2))      # new value of y
                    zk = a[2] + (h[2] * ((2 * m - 1) / 2))      # new value of z
                    k.append(integrand(xi, yj, zk))

        k = np.sum(k)                                       # Sum all values in k
        i2 = (1/2) * i1 + (1/2) * h[0]*h[1]*h[2] * k        # Update value of the integral
        h = h/2                                             # Half the step size
        c += 1                                              # Increase counter by 1
    samples = (2 **(c - 1) + 1)**3                          # Total Samples in all dimensions
    return i2[0], samples


def simpson_method3d(a, b, e, integrand):

    """
    :param a: lower limit of the integral, an array of x,y,z for each dimension
    :param b: upper limit of the integral,  an array of x,y,z for each dimension
    :param e: the user-specified relative accuracy
    :param integrand: A pointer which should have the name of the integrand function to use
    :return s: Returns the value of the integral using the simpson method in three dimensions

    This method uses the extended simpson method of integration to evaluate a 3 dimensional supplied integrand
    between the limits a and b of each dimension
    """
    if (b <= a).any():
        raise ValueError("b must be greater than a to evaluate integral")
    if np.size(b) !=3 :
        raise ValueError("b must have the dimensions 1 x 3 as an array")
    if np.size(a) !=3 :
        raise ValueError("a must have the dimensions 1 x 3 as an array")
    h = b - a                                   # Length of each coordinate system
    hi = np.average(h)                          # average length for initial testing
    i = []
    s_l = []
    i1 = 0.5 * hi * (integrand(a[0], a[1], a[2]) + integrand(b[0], b[1], b[2])) # first volume
    i.append(i1)
    mid = 0.5 * (a + b)                         # find midpoint of each coordinate system
    middle = integrand(mid[0], mid[1], mid[2])  # evaluation of each midpoint
    hi = hi / 2
    i2 = hi * (0.5 * integrand(a[0], a[1], a[2]) + 0.5 * integrand(b[0], b[1], b[2]) + middle)  # volume at n = 2
    i.append(i2)
    c = 2  # counter value, which takes into account the two iterations just done
    s = (4/3)*i2 - (1/3) * i1               # simpson method part
    s_l.append(i2)
    s_l.append(s)
    h = h/2
    while abs((s_l[1] - s_l[0]) / s_l[0]) > e:  # Keep iterating until relative error is below user specified
        i[0] = i[1]
        s_l[0] = s_l[1]
        k = []  # storage of the summation in the formula

        for l in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the x domain
            for j in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the y domain
                for m in range(1, 2**(c-1) + 1):    # Iterate through incrementing powers of 2 for the z domain
                    xi = a[0] + (h[0] * ((2*l - 1)/2))          # new value of x
                    yj = a[1] + (h[1] * ((2 * j - 1) / 2))      # new value of y
                    zk = a[2] + (h[2] * ((2 * m - 1) / 2))      # new value of z
                    k.append(integrand(xi, yj, zk))
        k = np.sum(k)  # Sum all values in k
        i[1] = (1 / 2) * i[0] + (1 / 2) * h[0]*h[1]*h[2] * k  # Update value of the integral at position 1 in the list
        s1 = (4/3)*i[1] - (1/3) * i[0]            # calculate the simpson value using the two integral
        s_l[1] = s1                               # update the value
        h = h / 2  # Half the step size
        c += 1  # Increase counter by 1
    s_l = s_l[1]
    samples = (2 **(c - 1) + 1) ** 3
    return s_l[0], samples

def main():
    # Validation of methods for 1D methods

    # validation 1
    "Validation of the 1D Integration methods using valid1, which has an analytical solution of 3250 when integrated "\
        " between a = 1 and b = 6"

    Int_trap_valid1, eval_trap_1D = trapezoidal_method(1, 6, 0.001, valid1)
    Int_simp_valid1, eval_simp_1D = simpson_method(1, 6, 0.00511, valid1)
    print("The evaluation of the integrand in valid1 using the trapezoidal rule is: " + str(Int_trap_valid1) +
          " The number of evaluations required is: " + str(eval_trap_1D))
    print("The evaluation of the integrand in valid1 using the Simpson rule is: " + str(Int_simp_valid1) +
          " The number of evaluations required is: " + str(eval_simp_1D))
    print(" ")

    # validation 2
    " Validation of the 1D Integration methods using valid2, which has an analytical solution of 1 when #"
    "integrated between a = 0 and b = pi/2"

    Int_trap_valid2, eval_trap_1D = trapezoidal_method(0, np.pi/2, 0.0000001, valid2)
    Int_simp_valid2, eval_simp_1D = simpson_method(0, np.pi/2, 0.0000001, valid2)
    print("The evaluation of the integrand in valid2 using the trapezoidal rule is: " + str(Int_trap_valid2) +
          " The number of evaluations required is: " + str(eval_trap_1D))
    print("The evaluation of the integrand in valid2 using the Simpson rule is: " + str(Int_simp_valid2) +
          " The number of evaluations required is: " + str(eval_simp_1D))
    print(" ")


    # Validation of methods for 3D methods
    # validation 3
    " Validation of the 3D Integration methods using valid3, which has an analytical solution of 1 when integrated between"\
        " a = [0,0,0] and b = [1,1,1]"

    a = np.array([[0], [0], [0]])
    b = np.array([[1], [1], [1]])

    Int_trap_3D_v3, eval_trap_3D_v3 = trapezoidal_method3d(a, b, 0.01, valid3)
    Int_simp_3D_v3, eval_simp_3D_v3 = simpson_method3d(a, b, 0.01, valid3)
    print("The evaluation of the integrand in valid3 using the 3D trapezoidal rule is: " +str(Int_trap_3D_v3)
          + " The number of evaluations required is: " + str(eval_trap_3D_v3))
    print("The evaluation of the integrand in valid3 using the 3D simpson rule is: " + str(Int_simp_3D_v3) +
          " The number of evaluations required is: " + str(eval_simp_3D_v3))
    print(" ")

    # validation 4
    " Validation of the 3D Integration methods using valid4, which has an analytical solution of -18 when integrated"\
        "  between a = [-1,0,-3] and b = [2,1,1]"

    a = np.array([[-1], [0], [-3]])
    b = np.array([[2], [1], [1]])

    Int_trap_3D_v4, eval_trap_3D_v4 = trapezoidal_method3d(a, b, 0.1, valid4)
    Int_simp_3D_v4, eval_simp_3D_v4 = simpson_method3d(a, b, 0.01, valid4)
    print("The evaluation of the integrand in valid4 using the 3D trapezoidal rule is: " +str(Int_trap_3D_v4)
          + " The number of evaluations required is: " + str(eval_trap_3D_v4))
    print("The evaluation of the integrand in valid4 using the 3D simpson rule is: " + str(Int_simp_3D_v4) +
          " The number of evaluations required is: " + str(eval_simp_3D_v4))
    print(" ")


    # Evaluation of the  of the 1-D simple harmonic oscillator potential the ground state time-independent wave function
    Int_trap_1D, eval_trap_1D = trapezoidal_method(0, 2, 0.000001, simple_harmonic)
    Int_simp_1D, eval_simp_1D = simpson_method(0, 2, 0.000001, simple_harmonic)
    print("The Probability of finding the particle for 1D wave function using trapezoidal rule is: " + str(Int_trap_1D) +
          " The number of evaluations required is: " + str(eval_trap_1D))
    print("The Probability of finding the particle for 1D wave function using simpson rule is: " + str(Int_simp_1D) +
          " The number of evaluations required is: " + str(eval_simp_1D))
    print(" ")


    # Evaluation of the  of the 3D ground state wave function to obtain the probability of finding a particle
    a = np.array([[0], [0], [0]])
    b = np.array([[2], [2], [2]])
    Int_trap_3D_g, eval_trap_3D_g = trapezoidal_method3d(a, b, 0.01, ground_state3d)
    Int_simp_3D_g, eval_simp_3D_g = simpson_method3d(a, b, 0.01, ground_state3d)
    print("The Probability of finding the particle for 3D ground state using the trapezoidal rule is: " + str(Int_trap_3D_g)
          + " The number of evaluations required is: " + str(eval_trap_3D_g))
    print("The Probability of finding the particle for 3D ground state using simpson rule is: " + str(Int_simp_3D_g) +
          " The number of evaluations required is: " + str(eval_simp_3D_g))
    print(" ")

    # Evaluation of the  of the 3D first state wave function to obtain the probability of finding a particle
    a = np.array([[0], [0], [0]])
    b = np.array([[2], [2], [2]])

    Int_trap_3D_1, eval_trap_3D_1 = trapezoidal_method3d(a, b, 0.01, first_state3d)
    Int_simp_3D_1, eval_simp_3D_1 = simpson_method3d(a, b, 0.01, first_state3d)
    print("The Probability of finding the particle for 3D first using the trapezoidal rule is: " + str(Int_trap_3D_1)
          + " The number of evaluations required is: " + str(eval_trap_3D_1))
    print("The Probability of finding the particle for 3D first state using simpson rule is: " + str(Int_simp_3D_1) +
          " The number of evaluations required is: " + str(eval_simp_3D_1))
    print(" ")

if __name__ == "__main__":
    main()

