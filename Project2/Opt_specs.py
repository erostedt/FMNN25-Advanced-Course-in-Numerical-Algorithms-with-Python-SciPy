import numpy as np
import math
import scipy as sp


class Hessian:
    """
    Class that only contains class methods that use different methods to compute the Hessian matrix
    """
    @classmethod
    def finite_diff(cls, f, x):
        """
        Calculates Hessian using by taking symmetric steps.
        :param f: Function f.
        :param x: Current point x.
        :return: Approximate hessian matrix.
        """
        h = 1e-8
        n = len(x)
        G = np.zeros((n, n))
        fx = f(x)
        denum = h*h
        for r in range(n):
            r_step_vector = np.zeros(n)
            r_step_vector[r] = h
            for c in range(n):
                c_step_vector = np.zeros(n)
                c_step_vector[c] = h
                G[r, c] = (f(x + r_step_vector + c_step_vector) - f(x + r_step_vector) - f(x + c_step_vector) + fx)/denum
        return (G + G.T)/2

    @classmethod
    def broyden_good(cls, H_k, delta_k, gamma_k):
        """
        Class method for "good" Broyden method for inverse Hessian matrix
        :param H_k: The previous Hessian matrix
        :param delta_k: current point - previous point
        :param gamma_k: current gradient - previous gradient
        :returns: The new inverse Hessian matrix
        """
        u = delta_k - H_k @ gamma_k
        a = 1 / (u.T @ gamma_k)
        H_k1 = H_k + a * (u @ u.T)
        return H_k1

    @classmethod
    def broyden_bad(cls, G_k, delta_k, gamma_k):
        """
        Class method for "bad" Broyden method for Hessian matrix
        :param H_k: The previous Hessian matrix
        :param delta_k: current point - previous point
        :param gamma_k: current gradient - previous gradient
        :returns: The new Hessian matrix
        """
        G_k1 = G_k+(gamma_k-G_k @ delta_k)/(delta_k.T @ delta_k) @ delta_k.T
        return G_k1

    @classmethod
    def DFP(cls, H_k, delta_k, gamma_k):
        """
        Class method for Davidson-Fletcher-Powell method for Hessian matrix
        :param H_k: The previous Hessian matrix
        :param delta_k: current point - previous point
        :param gamma_k: current gradient - previous gradient
        :returns: The new inverse Hessian matrix
        """
        H_kp1 = H_k + (delta_k @ delta_k.T)/(delta_k.T @ gamma_k) - \
            ((H_k @ gamma_k) @ (gamma_k.T @ H_k))/(gamma_k.T @ H_k @ gamma_k)
        return H_kp1

    @classmethod
    def BFGS(cls, H_k, delta_k, gamma_k):
        """
        Class method for Broyden-Fletcher-Goldfarb-Shanno method for Hessian matrix
        :param H_k: The previous Hessian matrix
        :param delta_k: current point - previous point
        :param gamma_k: current gradient - previous gradient
        :returns: The new inverse Hessian matrix
        """
        term1 = 1 + ( gamma_k.T @ H_k @ gamma_k ) / ( delta_k.T @ gamma_k )
        term2 = ( delta_k @ delta_k.T ) / ( delta_k.T @ gamma_k )
        term3 = ( delta_k @ (gamma_k.T @ H_k) + H_k @ (gamma_k @ delta_k.T)) / \
            ( delta_k.T @ gamma_k )

        H_kp1 = H_k + term1 * term2 - term3
        return H_kp1


class LineSearch:
    """
    Class with only class methods that specifies different types of line search methods
    """

    @classmethod
    def exact_search(cls, f, g, x0, d):
        """
        Line search method based on golden section search.
        :param f: Function f.
        :param g: gradient g.
        :param x0: current point x.
        :param d: direction d.
        :return: Step length.
        """
        fd = (lambda a: f(x0+a*d))

        val1 = fd(0)
        startval = val1
        right = 1
        l = 1
        while fd(l) <= val1:
            val1 = fd(l)
            l *= 10
        while fd(l) > startval:
            l /= 1.5
        alpha = (-1 + math.sqrt(5)) / 2
        left = 0
        right = l
        b1 = right
        l = left + (1 - alpha) * (right - left)
        m = left + alpha * (right - left)
        val1 = fd(l)
        val2 = fd(m)

        while right != 0 and (abs((left-right)/right) > 1e-9 or abs(val1-val2) > 1e-9):

            if val1 > val2:
                left = l
                l = m
                m = left + alpha * (right - left)
                val1 = val2
                val2 = fd(m)
            else:
                right = m
                m = l
                l = left + (1 - alpha) * (right - left)
                val2 = val1
                val1 = fd(l)
        l=(right+left)/2
        if fd(l)>fd(b1):
            l=b1
        if fd(l)>fd(right):
            l=right
        return l

    @classmethod
    def inexact_search(cls, f, g, x, d, param=(0.1,0.7, 0.1, 9)):
        """
        Inexact line search, picks first acceptable point it finds (Acceptable points based on Wolfe conditions)
        :param f: Function f.
        :param g: gradient g
        :param x: current point.
        :param d: Direction d.
        :param param: (Optional) Parameters for the constants rho, sigma, tau, xi.
        :return: Step length a0 and function value at x+ad (i.e, f(x+ad)).
        """
        if param[0] >= param[1]:
            raise InputError("param[0] >= param[1]", "rho may not be larger than sigma")

        rho, sigma, tau, xi = param[0], param[1], param[2], param[3]
        a_lower, a0, a_upper = 0, 1, 1e99

        fa0 = f(x + a0 * d)
        fa_lower = f(x + a_lower)

        if type(x) is int:
            f_prime = d * g(x + d * a0)
            f_prime_lower = d * g(x + d * a_lower)
        else:
            f_prime = d @ g(x + d * a0)
            f_prime_lower = d @ g(x + d * a_lower)

        LC = f_prime >= sigma*f_prime_lower
        RC = fa0 <= fa_lower + rho*(a0-a_lower)*f_prime_lower

        # Wolfe conditions
        while not (LC and RC):
            if f_prime_lower==f_prime:
                return a0
            if not LC:
                da0 = (a0 - a_lower) * f_prime / (f_prime_lower - f_prime)
                da0 = max(da0, tau * (a0 - a_lower))
                da0 = min(da0, xi * (a0 - a_lower))
                a_lower = a0
                a0 = a0 + da0
            else:
                a_upper = min(a0, a_upper)
                a_bar = (a0 - a_lower) ** 2 * f_prime_lower / (2 * (fa_lower - fa0 + (a0 - a_lower) * f_prime_lower))
                a_bar = max(a_bar, a_lower + tau * (a_upper - a_lower))
                a_bar = min(a_bar, a_upper - tau * (a_upper - a_lower))
                a0 = a_bar

            fa0 = f(x + a0 * d)
            fa_lower = f(x + a_lower)
            if type(x) is int:
                f_prime = d * g(x + d * a0)
                f_prime_lower = d * g(x + d * a_lower)
            else:
                f_prime = d @ g(x + d * a0)
                f_prime_lower = d @ g(x + d * a_lower)

            LC = f_prime >= sigma * f_prime_lower
            RC = fa0 <= fa_lower + rho * (a0 - a_lower) * f_prime_lower

        return a0  # , fa0


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
