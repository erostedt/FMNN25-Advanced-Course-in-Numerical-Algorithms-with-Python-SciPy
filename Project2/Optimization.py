import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import Opt_specs
from Opt_specs import Hessian


class OptimizationProblem:
    """
    Class to specify the given optimization problem. To be used with a solver class.
    """

    def __init__(self, dim, function, gradient=False):
        """
        :param dim: The dimension of the argument in the optimizationproblem
        :param function: A function f:R^dim-->R to be minimized
        :param gradient: Optionally the gradient to function can be given
        """
        self.dim = dim
        self.f = function
        self.given_gradient = False
        if gradient:
            self.given_gradient = True
            self.gradient = gradient

    def f(self, x):
        """
        Function that returns f(x).
        :param x: Current point x.
        :return: f(x)
        """
        return self.f(x)

    def grad(self, x):
        """
        If gradient was given, use that one. If gradient was not given, we calculate our own.
        :param x: Current point x.
        :return: Gradient g(x).
        """
        if self.given_gradient:
            return self.gradient(x)
        else:
            basis = np.eye(self.dim)
            eps = 1e-7
            grad = np.array([(self.f(x+eps*max(abs(x[i]), 0.1)*basis[:, i])-self.f(x))\
                /(eps*max(abs(x[i]), 0.1)) for i in range(self.dim)])
            return grad

    def plot(self, x0_range, x1_range, logspace=False, show=True):
        """
        Plots contours of a function such as Rosenbrock
        :param x0_range: x range on which to plot
        :param x1_range: y range on which to plot
        :param logspace: if plot is to be logarithmic (default: false)
        :param show: if plot is to be shown directly, (default: true)
        """
        z = [self.f([x0, x1]) for x1 in x1_range for x0 in x0_range]
        if logspace:
            plt.contour(x0_range, x1_range, np.reshape(z, (-1, len(x1_range))),
                        np.logspace(-0.5, 3.5, 20, base=10))
        else:
            plt.contour(x0_range, x1_range, np.reshape(z, (-1, len(x1_range))))
        if show:
            plt.show()


class Optimization:
    def Find_Minimum(self, x0, tol=1e-15, gradtol=1e-3, plots=False, plotHess=False):
        """
        Find minimum of function.
        :param x0: Initial point x0.
        :param tol: Tolerance tol of the 2-norm of the difference between
                    iterations, default 1e-10.
        :param gradtol: Tolerance for gradient.
        :param plots: True if one wishes to see plot.
        :param plotHess: True if one wishes to see plot of hessian.
        :return: Minimum position and minimum value.
        """

        #create list to plot iterations
        self.plotpoints = [x0]
        x_k = x0
        xdiff = tol+1
        H_k=np.eye(len(x0))
        
        #create list to plot invhessian approx
        self.Hesspoints = [np.linalg.norm(H_k-Opt_specs.Hessian.finite_diff(
            self.optimization_problem.f, x0))]
        
        #Starting with H_k=I leads to problems if classical newton method is used
        if self._who_am_I() == "ClassicalNewtonNoLine":
            H_k = self._ClassicalHessian(self.optimization_problem.f, x_k)        
        g_k = self.optimization_problem.grad(x0)
        
        while xdiff > tol or np.linalg.norm(g_k) > gradtol:
            #Newton direction
            s_k = self._NewtonDirection(H_k, g_k)
            #Line search to find minimum ni newton direction, or step 1 if classical newton is used
            if self._who_am_I() == "ClassicalNewtonNoLine":
                alpha_k = 1
            else: alpha_k = self._LineSearch(self.optimization_problem.f, self.optimization_problem.grad, x_k, s_k)
            
            if alpha_k < 1e-15 or np.linalg.norm(s_k) < 1e-15:
                break
            
            x_k1 = x_k + alpha_k * s_k
            #save iterations if one wishes to
            if plots:
                self.plotpoints.append(x_k1)
            g_k1 = self.optimization_problem.grad(x_k1)
            delta_k = alpha_k * s_k
            gamma_k = g_k1 - g_k
            gamma_k = gamma_k.reshape(len(gamma_k), 1)
            delta_k = delta_k.reshape(len(gamma_k), 1)

            # Choose method to update Hessian (abstract)
            if self._who_am_I() == "ClassicalNewtonLine" or \
                    self._who_am_I() == "ClassicalNewtonNoLine":
                H_k = self._ClassicalHessian(self.optimization_problem.f, x_k1)
            else:
                H_k = self._Hessian(H_k, delta_k, gamma_k)

            # Save invHess aproximation if one wishes to
            if plotHess:
                self.Hesspoints.append(np.linalg.norm(H_k - np.linalg.inv(
                    Opt_specs.Hessian.finite_diff(self.optimization_problem.f, x_k))))
            xdiff = np.linalg.norm((x_k - x_k1))
            
            #Choose method to update Hessian (abstract)
            if self._who_am_I() == "ClassicalNewtonLine" or \
                    self._who_am_I() == "ClassicalNewtonNoLine":
                H_k = self._ClassicalHessian(self.optimization_problem.f, x_k1)
            else: H_k = self._Hessian(H_k, delta_k, gamma_k)
            
            #Save invHess aproximation if one wishes to
            if plotHess:
                self.Hesspoints.append(np.linalg.norm(H_k-np.linalg.inv(
                    Opt_specs.Hessian.finite_diff(self.optimization_problem.f, x_k))))
                    #np.array([[1200*x_k1[0]-400*x_k1[1]+2,-400*x_k1[0]],[-400*x_k1[0],200]])))
            xdiff = np.linalg.norm((x_k-x_k1))
            x_k = x_k1
            g_k = g_k1
        return x_k, self.optimization_problem.f(x_k)

    def _Hessian(self, H_k, delta_k, gamma_k, hessian_method):
        """
        Abstract method for computing the Hessian matrix
        :param H_k: Hessian matrix of last step
        :param delta_k: delta_k is defined as x_k - x_kn1
        :param gamma_k: gamma_k is defined as g_k - g_kn1 where g is the gradient
        :param hessian_method: Specifies which method to be used when computing the next Hessian
        :returns: the next Hessian matrix
        """
        pass

    def _NewtonDirection(self, H_k, g_k):
        """
        Abstract method for specifying the Newton direction
        :param H_k: the Hessian matrix
        :param g_k: the corresponding gradient
        :returns: the Newton direction
        """
        pass

    def _ClassicalHessian(self, function, x_value):
        """
        Abstract method for using the classical Hessian method
        :param function: the function to be optimized
        :param x_value: the x value
        :returns: the next Hessian
        """
        pass

    def _who_am_I(self):
        """
        Method that returns which type the solver is
        :return: class name
        """
        return type(self).__name__

    def plot_solution(self, x0_range, x1_range):
        """
        Plots the solution
        :param x0_range: x0 range on which to plot
        :param x1_range: x1 range on which to plot
        """
        self.optimization_problem.plot(x0_range, x1_range, logspace=True, show=False)
        plt.plot(*zip(*self.plotpoints), '-ro')
        plt.show()

    def plot_Hessapprox(self):
        """
        Plotting of fault of Hessian approximation
        """
        plt.scatter(range(len(self.Hesspoints)), self.Hesspoints)
        plt.gca().set_yscale('log')
        plt.gca().set_ylim(bottom=10**-3)
        plt.show()

    def _LineSearch(self, function, grad, x_value, s_k):
        """
        Line search to find step length.
        :param function: Function function.
        :param grad: Gradient grad
        :param x_value: Current point x_value.
        :param s_k: Search direction s_k.
        :return: Step length.
        """
        if self.exact_ls:
            return Opt_specs.LineSearch.exact_search(function, grad, x_value, s_k)
        return Opt_specs.LineSearch.inexact_search(function, grad, x_value, s_k)


class ClassicalNewtonNoLine(Optimization):
    """
    Class that inherits the solver class Optimization and specifies Newton method
    for what is to be used when solving the optimization problem
    """
    def __init__(self, optimization_problem):
        """
        :param optimization_problem: Which problem to optimize
        """
        self.optimization_problem = optimization_problem

    def _ClassicalHessian(self, function, x_value):
        """
        Calls for finite difference Hessian.
        :param function: Function f.
        :param x_value: Current point x.
        :return: approximate Hessian matrix.
        """
        return Hessian.finite_diff(function, x_value)

    def _NewtonDirection(self, H_k, g_k):
        """
        Specifies the Newton direction
        :param H_k: the Hessian matrix
        :param g_k: the corresponding gradient
        :returns: the Newton direction
        """
        return -1*solve(H_k, g_k)


class ClassicalNewtonLine(ClassicalNewtonNoLine):
    """
    Class that inherits the solver class Optimization and specifies methods
    for what is to be used when solving the optimization problem
    """

    def __init__(self, optimization_problem, exact_ls=True):
        """
        :param optimization_problem: Which problem to optimize
        :param exact_LS: if true, exact Line search will be used. Default: true.
        """
        self.optimization_problem = optimization_problem
        self.exact_ls = exact_ls


class QuasiNewton(Optimization):
    """
    Class that inherits the solver class Optimization and specifies Quasi Newton methods
    for what is to be used when solving the optimization problem
    """
    def __init__(self, optimization_problem, exact_ls=True, hessian_method='BFGS'):
        """
        :param optimization_problem: An optimizationProblem to be minimized
        :param exact_LS: specifies if exact linesearch is to be used
        :param hessian_method: Specifies what method to use when updating the aproximate inverse hessian
                Set as BFGS as default
                Can choose between: BFGS (default, Broyden-Fletcher-Goldfarb-Shanno), 
                BB (Broyden bad), BG (Broyden good),
                DFP (Davidson-Fletcher-Powell method)
        """
        self.optimization_problem = optimization_problem
        self.exact_ls = exact_ls
        self.hessian_method = hessian_method

    def _Hessian(self, H_k, delta_k, gamma_k):
        """
        Choice of Hessian.
        :param H_k: The previous Hessian matrix
        :param delta_k: current point - previous point
        :param gamma_k: current gradient - previous gradient
        :return: Approximate Hessian inverse matrix.
        """
        if self.hessian_method == 'BG': return Hessian.broyden_good(H_k, delta_k, gamma_k)
        elif self.hessian_method == 'BB': return Hessian.broyden_bad(H_k, delta_k, gamma_k)
        elif self.hessian_method == 'DFP': return Hessian.DFP(H_k, delta_k, gamma_k)
        elif self.hessian_method == 'BFGS': return Hessian.BFGS(H_k, delta_k, gamma_k)
        else: raise NameError('Invalid name of Hessian method!')

    def _NewtonDirection(self, H_k, g_k):
        """
        Specifies the Newton direction
        :param H_k: the inverse Hessian, or Hessian matrix, depending on method
        :param g_k: the corresponding gradient
        :returns: the Newton direction
        """
        if self.hessian_method == 'broyden bad':
            return -1*solve(H_k, g_k)
        return -1*H_k @ g_k 

