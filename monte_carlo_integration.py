import numpy as np
import matplotlib.pyplot as plt



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


def valid3(x):
    """

    :param x: The array of x,y and z coordinates to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

    This function has an analytical solution of 1 when evaluated between a = [0,0,0] and b = [1,1,1] and it can be
    used to verify and validate the correct implementation of the 3D monte carlo method.
    """

    return x[0]**2 + x[1]**2 + x[2]**2


def valid4(x):
    """

    :param x: The array of x,y and z coordinates to evaluate the function
    :return:  Returns the evaluation of the integral using the three coordinates

    This function has an analytical solution of -18 when evaluated between a = [0,-1,-3] and b = [1,2,1] and it can be
    used to verify and validate the correct implementation of the 3D monte carlo method.
    """

    return 6*x[0]*x[1]*x[2]


# Following are functions which needs to be integrated to obtain the respective probabilities, as per the tasks
def simple_harmonic(x):

    """
    :param x: Takes in values of x supplied from the integration function
    :return: Returns the evaluation of the function at x

       This function calculates the ground state of a 1D harmonic oscillator squared, and is then integrated to
       calculate the probability
    """

    return np.sqrt(1 / np.pi) * np.exp(-x ** 2)


def ground_state3d(x):

    """
    :param x: Takes in three dimensional array for the coordinates, x, y, z
    :return: Returns the evaluation of the function at x

       This function calculates the ground state of a 1D harmonic oscillator squared, and is then integrated to
       calculate the probability
    """
    c = np.sqrt(1 / np.pi)
    return c*c*c * np.exp(-x[0] ** 2)*np.exp(-x[1] ** 2)*np.exp(-x[2] ** 2)


def first_state3d(x):
    """
       :param x: Takes in three dimensional array for the coordinates, x, y, z
       :return: Returns the evaluation of the function at x

          This function calculates the first state of a 1D harmonic oscillator squared, and is then integrated to
          calculate the probability
       """
    return (1/np.pi)**(3/2)*(x[0]**2 + x[1]**2)*np.exp(-(x[0]**2 + x[1]**2 + x[2]**2))


def pdf_random(x):
    """

    :param x: Input x which is random value between 0 and 1
    :return: A  value which is generated between 0 and 1 that has been adjusted to a new distribution, which is
    the inverse of the integral of the  PDF function provided for importance sampling
    """
    a = -1/1.6          # the gradient which has been normalised
    b = 2.1/1.6         # the y intercept which has been normalised
    k = -(b / a) - np.sqrt((2 * x * a + b ** 2) / a ** 2)       # the equation of the new distribution
    return k


def pdf(x):
    """
    :param x:
    :return: Evaluation of x using the formula ax + b
    """

    a = -1/1.6  # normalised gradient
    b = 2.1/1.6 # normalised y intercept
    return a*x + b


# Following is the 1D monte carlo method
def monte_carlo_func(a, b, e, integrand, sampling):
    """
    :param a: lower limit of the integral
    :param b: upper limit of the integral
    :param e:   accuracy
    :param integrand: the function to integrate
    :param sampling: The sampling type, 'flat' = no sampling, '1' = sampling with pdf_random function

    :return integrals[1] : evaluation of the integrand
    :return adj_array : Array of samples taken in the space
    :return error : Error associated with the method
    :return n : Number of samples taken

    This function uses the monte carlo method of integration, to integrate a supplied integrand with a specified
    type of sampling to a relative accuracy
    """
    if b <= a:
        raise ValueError("b must be greater than a to evaluate integral")
    global adj_array
    sam_num = np.array([10000, 20000])                  # initial number of samples
    integrals = np.zeros(2)                     # integral holder
    vol = b - a                                 # volume / limits
    n = 20000
    inc_size = 100000

    for i in range(2):                          # find first two integrals
        ran_array = np.random.random(sam_num[i])   # create an array of random variables between 0 and 1, of sample size
        if sampling == 'flat':                  # specify the type of sampling
            adj_array = (a + vol * ran_array)   # scale the random variables
            p = 1                               # pdf variable, for no sampling it is 1
        if sampling == '1':                     # sampling using the pdf and pdf random functions
            r_new = pdf_random(ran_array)
            adj_array = (a + vol * r_new)       # scale variable
            p = pdf(r_new)                      # new random variables that have been transformed
        int_array = integrand(adj_array)/p
        val1 = np.sum(int_array)    # the summation of all evaluation's
        integrals[i] = ((vol * val1) / sam_num[i])   # update the two initial integrals
        sig_f = np.sqrt(1 / (sam_num[i] - 1) * np.sum((int_array - integrals[i]) ** 2))  # error of the function
        error = (vol * sig_f) / np.sqrt(sam_num[i])  # error of the integral
        inn_error = np.sum((int_array - integrals[i]) ** 2)

    while abs(error / integrals[1]) > e:    # iterate until relative accuracy achieved

        val = val1
        integrals[0] = integrals[1]             # update the previous integral
        ran_array = np.random.random(inc_size)    # create random array with a size of the given sample number
        np.random.shuffle(ran_array)
        if sampling == 'flat':                  # specify the type of sampling
            adj_array = (a + vol * ran_array)   # scale the random variable
            p = 1

        if sampling == '1':                     # specify the sampling, quantum sampling
            r_new = pdf_random(ran_array)
            adj_array = (a + vol * r_new)       # scale the random variable
            p = pdf(r_new)
        int_array = integrand(adj_array)/p
        val1 = val + np.sum(int_array)
        integrals[1] = (val1 * vol) / (n + inc_size)         # update the last integral
        n += inc_size                                           # Increment the size of n
        inn_error += np.sum((int_array - integrals[1]) ** 2)    # Running total of the sum for the error
        sig_f = np.sqrt((1 / (n - 1)) * inn_error)              # Error
        if sampling == 'flat':                                  # Error for flat sampling
            error = (vol * sig_f) / np.sqrt(n)
        if sampling =='1':                                      # Error for importance sampling
            error = (sig_f) / np.sqrt(n)

    return integrals[1], adj_array, error/integrals[1], n


# Following is the 3D monte carlo method
def three_dimen_monte(a, b, e, integrand, sampling, dimension):
    """
    :param a: array of lower limits for x,y,z coordinates
    :param b: array of upper limits for x,y,z coordinates
    :param e:  Accuracy
    :param integrand: the function to integrate
    :param sampling: The sampling type, 'flat' = no sampling, '1' = sampling with pdf_random function
    :param dimension : The dimension to integrate over, as 1D monte carlo method can also be done in this func
                        Must be the same as the provided a and b arrays

    :return integrals[1] : evaluation of the integrand
    :return error : Error associated with the method
    :return n : Number of samples taken


    This function uses the monte carlo method of integration, to integrate a three dimension function.
    """
    if (b <= a).any():
        raise ValueError("b must be greater than a to evaluate integral")
    if np.size(a) != dimension :
        raise ValueError("a must have the same dimensions as specified in the function, i.e 1 x dimension")
    if np.size(b) != dimension :
        raise ValueError("b must have the same dimensions as specified in the function, i.e 1 x dimension")
    global adj_array

    sam_num = np.array([10000, 20000])                # initial number of samples, to create first two integrals
    integrals = np.zeros(2)                     # integral holder
    length_arr = b - a                          # length of each axis
    vol = np.product(length_arr)                # volume of the integral
    n = 20000                                       # initial sample number for loop
    inc_size = 100000                           # increment size for each iteration in while loop

    for i in range(2):                          # find first two integrals
        ran_array = np.random.random((dimension, sam_num[i]))  # create 3d array of random numbers between 0 and

        if sampling == 'flat':                  # specify the type of sampling
            adj_array = (a + length_arr * ran_array)   # scale the random variables
            p = 1                               # pdf variable, for no sampling it is 1
        if sampling == '1':                     # sampling using the pdf and pdf random functions
            r_new = pdf_random(ran_array)
            adj_array = (a + length_arr * r_new)       # scale variable
            p = pdf(r_new)                      # new random variables that have been transformed
            p = np.product(p, axis = 0)
        int_array = integrand(adj_array)/p
        val1 = np.sum(int_array)    # the summation of all evaluation's
        integrals[i] = ((vol * val1) /(sam_num[i]))   # update the two initial integrals
        sig_f = np.sqrt(1 / (sam_num[i] - 1) * np.sum((int_array - integrals[i]) ** 2))  # error of the function
        error = (vol * sig_f) / np.sqrt(sam_num[i])  # error of the integral
        inn_error = np.sum((int_array - integrals[i]) ** 2)
    while abs(error /integrals[1]) > e:    # iterate until relative accuracy achieved
        val = val1
        integrals[0] = integrals[1]             # update the previous integral
        ran_array = np.random.random((dimension,inc_size))  # create random array with a size of the given sample number
        np.random.shuffle(ran_array)
        if sampling == 'flat':                  # specify the type of sampling
            adj_array = (a + length_arr * ran_array)   # scale the random variable
            p = 1
        if sampling == '1':                     # specify the sampling, quantum sampling
            r_new = pdf_random(ran_array)
            adj_array = (a + length_arr * r_new)       # scale the random variable
            p = pdf(r_new)
            p = np.product(p, axis=0)
        int_array = integrand(adj_array)/p              # vectorised the inner part of sum
        val1 = val + np.sum(int_array)
        integrals[1] = (val1 * vol) / (n + inc_size)          # update the last integral
        n += inc_size                                         # Increase value of n
        inn_error += np.sum((int_array-integrals[1])**2)      # Updating error

        sig_f = np.sqrt((1/(n-1))*inn_error)                  # Error
        if sampling == 'flat':
            error = (vol * sig_f) / np.sqrt(n)
        if sampling == '1':
            error = (sig_f) / np.sqrt(n)

    return integrals[1], error/integrals[1], n

def main():

    # Validation of methods for 1D methods
    # validation 1
    "Validation of the 1D monte carlo method using valid1, which has an analytical solution of 3250 when integrated "\
        " between a = 1 and b = 6"
    int_1D_monte_flat, samp_val1, error_1D_monte1, number_samples1 = monte_carlo_func(1, 6, 0.001,valid1, 'flat')

    print("The evaluation of the integrand in valid1 using the 1D monte carlo method with no sampling is: "
          + str(int_1D_monte_flat) + " The relative error in the simulation is: " + str(error_1D_monte1) +
          " The number of samples required is: " + str(number_samples1))

    print(" ")

    # validation 2
    " Validation of the 1D of monte carlo method using valid2, which has an analytical solution of 1 when integrated " \
        "between a = 0 and b = pi/2"

    int_1D_monte_flat_v2, samp_val_v2, error_1D_monte_v2, number_samples_v2 = monte_carlo_func(0, np.pi/2, 0.0001,valid2,
                                                                                               'flat')
    int_1D_monte_samp_v2, samp_val__s_v2, error_1D_monte_s__v2, number_samples_s_v2 = monte_carlo_func(0, np.pi/2, 0.0001
                                                                                                       , valid2, '1')

    print("The evaluation of the integrand in valid2 using the 1D monte carlo method with no sampling is: "
          + str(int_1D_monte_flat_v2) + " The relative error in the simulation is: " + str(error_1D_monte_v2) +
          " The number of samples required is: " + str(number_samples_v2))
    print("The evaluation of the integrand in valid2 using the 1D monte carlo method with sampling is: "
          + str(int_1D_monte_samp_v2) + " The relative error in the simulation is: " + str(error_1D_monte_s__v2) +
          " The number of samples required is: " + str(number_samples_s_v2))

    print(" ")

    "The following is a plot to see how the number of samples square rooted changes depending on the relative accuracy"
    acc = [0.1,0.01,0.001,0.0001, 0.00001]
    sample_num = [316.2593872124589, 316.2593872124589, 774.6095790783896, 7483.316109853973, 74799.73275353329]
    # for i in acc:
    #     int_1D_monte_flat_v2, samp_val_v2, error_1D_monte_v2, number_samples_v2 = monte_carlo_func(0, np.pi / 2, i,
    #                                                                                                valid2, 'flat')
    #     sample_num.append(np.sqrt(number_samples_v2)) # Code to generate the above values. No need to uncomment
    plt.plot(acc, sample_num, marker = 'x', color = 'm')
    plt.xlabel("Relative error ", fontsize = 10)
    plt.ylabel("$N^{1/2}$", fontsize = 10)
    plt.show()
    # Validation of methods for 3D monte carlo
    # validation 3
    " Validation of the 3D monte carlo method using valid3, which has an analytical solution of 1 when integrated between"\
        " a = [0,0,0] and b = [1,1,1]"

    a = np.array([[0], [0], [0]])                   # Set up array of lower limit for the coordinates
    b = np.array([[1], [1], [1]])                   # set up array of the upper limit for the coordinates

    int_3D_monte_flat_v3,  error_3D_monte_v3, number_samples_V3 = three_dimen_monte(a, b, 0.0001, valid3,'flat', 3)

    print("The evaluation of the integrand in valid3 using the 3D monte carlo method with no sampling is: "
          + str(int_3D_monte_flat_v3) + " The relative error in the simulation is: " + str(error_3D_monte_v3) +
          " The number of samples required is: " + str(number_samples_V3))

    print("")

    # validation 4
    " Validation of the 3D Integration methods using valid4, which has an analytical solution of -18 when integrated"\
        "  between a = [-1,0,-3] and b = [2,1,1]"

    a = np.array([[-1], [0], [-3]])
    b = np.array([[2], [1], [1]])

    int_3D_monte_flat_v4,  error_3D_monte_v4, number_samples_V4 = three_dimen_monte(a, b, 0.01, valid4,'flat', 3)

    print("The evaluation of the integrand in valid4 using the 3D monte carlo method with no sampling is: "
          + str(int_3D_monte_flat_v4) + " The relative error in the simulation is: " + str(error_3D_monte_v4) +
          " The number of samples required is: " + str(number_samples_V4))
    print(" ")


    # Evaluation of the  of the 1-D simple harmonic oscillator potential the ground state time-independent wave function
    "The following is the plot of the 1D ground state system squared, which when integrated gives the probability of " \
        " of finding a particle within that domain"

    x_quantum_1D = np.linspace(0.00001,2,1000)
    y_quantum_1D = simple_harmonic(x_quantum_1D)
    Y_PDF = pdf(x_quantum_1D)

    plt.plot(x_quantum_1D, y_quantum_1D, label = "$\u0399\u03C8_{(x)}\u0399^2$, 1D Ground State Squared" )
    plt.plot(x_quantum_1D, Y_PDF, label = " Y = -0.625X + 1.3125, Sampling PDF ")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(b = bool )
    plt.legend(loc = "upper right")
    plt.show()

    "The following is the code to obtain the 1-Dimensional simple harmonic probability using the monte carlo method " \
        "with and without sampling "

    integral_1D_monte_sampling, sampled_values1, error_1D_monte1, number_samples1 = monte_carlo_func(0, 2, 0.0001,
                                                                                                  simple_harmonic, '1')
    integral_1D_monte_flat, sampled_values2, error_1D_monte2, number_samples2 = monte_carlo_func(0, 2, 0.0001,
                                                                                                  simple_harmonic, 'flat')
    print("The evaluation of the integrand in simple harmonic using the 1D monte carlo method with no sampling is: "
          + str(integral_1D_monte_flat) + " The relative error in the simulation is: " + str(error_1D_monte2) +
          " The number of samples required is: " + str(number_samples2))
    print("The evaluation of the integrand in simple harmonic using the 1D monte carlo method with sampling is: "
          + str(integral_1D_monte_sampling) + " The relative error in the simulation is: " + str(error_1D_monte1) +
          " The number of samples required is: " + str(number_samples1))
    plt.hist(sampled_values2)
    plt.xlabel("Flat Sampling Values")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(sampled_values1, color='m')
    plt.xlabel("Importance Sampled Values")
    plt.ylabel("Frequency")
    plt.show()
    print("")


    # Evaluation of the  of the 3D ground state wave function to obtain the probability of finding a particle
    "The following is a plot of the 3D ground state wave function. This plots x and y, with a constant z coordinate" \
        "I repeat this plot for different z values to show the layers"


    ax = plt.axes(projection='3d')
    x = np.linspace(0, 2)
    y = np.linspace(0, 2)
    x, y = np.meshgrid(x, y)
    z_list = []
    for i in range(9):
        z_list.append(ground_state3d([x,y,i/4]))

    ax.plot_surface(x, y, z_list[0], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[1], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[2], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[3], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[4], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[5], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[6], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[7], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    ax.plot_surface(x, y, z_list[8], rstride=1, cstride=1,
                cmap='magma', edgecolor='none')
    plt.show()

    "The following is the code to obtain the probability of the 3D ground state using the monte carlo method " \
        "with and without sampling "
    a = np.array([[0], [0], [0]])                   # Set up array of lower limit for the coordinates
    b = np.array([[2], [2], [2]])                   # set up array of the upper limit for the coordinates

    integral_3D_monte_flat_g,  error_3D_monte1, number_samples1 = three_dimen_monte(a, b, 0.001, ground_state3d,'flat', 3)
    integral_3D_monte_sample_g,  error_3D_monte2, number_samples2 = three_dimen_monte(a, b, 0.001, ground_state3d,'1', 3)
    print("The evaluation of the integrand in ground_state3d using the 3D monte carlo method with no sampling is: "
          + str(integral_3D_monte_flat_g) + " The relative error in the simulation is: " + str(error_3D_monte1) +
          " The number of samples required is: " + str(number_samples1))

    print("The evaluation of the integrand in ground_state3d using the 3D monte carlo method with sampling is: "
          + str(integral_3D_monte_sample_g) + " The relative error in the simulation is: " + str(error_3D_monte2) +
          " The number of samples required is: " + str(number_samples2))
    print("")



    # Evaluation of the  of the 3D first state wave function to obtain the probability of finding a particle
    "The following is the code to obtain the probability of the 3D first state using the monte carlo method " \
        "with and without sampling "
    a = np.array([[0], [0], [0]])                   # Set up array of lower limit for the coordinates
    b = np.array([[2], [2], [2]])                   # set up array of the upper limit for the coordinates

    integral_3D_monte_flat_1,  error_3D_monte1, number_samples1 = three_dimen_monte(a, b, 0.001, first_state3d,'flat', 3)
    integral_3D_monte_sample_1,  error_3D_monte2, number_samples2 = three_dimen_monte(a, b, 0.001, first_state3d,'1', 3)
    print("The evaluation of the integrand in first state using the 3D monte carlo method with no sampling is: "
          + str(integral_3D_monte_flat_1) + " The relative error in the simulation is: " + str(error_3D_monte1) +
          " The number of samples required is: " + str(number_samples1))
    print("The evaluation of the integrand in first state using the 3D monte carlo method with sampling is: "
          + str(integral_3D_monte_sample_1) + " The relative error in the simulation is: " + str(error_3D_monte2) +
          " The number of samples required is: " + str(number_samples2))
    print("")


if __name__ == "__main__":
    main()