import unittest
import Opt_specs
import Optimization
import numpy as np
import ChebyQuad_problem
import scipy.optimize as so


class Testing(unittest.TestCase):
# Testing of class LineSearch
    def test_Search(self):
        for a in [-4, -2, 2, 4]:
            func = (lambda x: (1-10**a*x)**2)
            grad = (lambda x: 2*(1-10**a*x)*-10**a)
            xs = 0
            d = 1
            xmin = Opt_specs.LineSearch.exact_search(func, grad, xs, d)
            xmin2 = Opt_specs.LineSearch.inexact_search(func, grad, xs, d)
            self.assertAlmostEqual(xmin, 1/10**a,3)
            self.assertTrue((xmin2 - 1 / 10**a) / (1/10 ** a) < 0.3)
    
    # Testing of class OptimizationProblem
    def testfg(self):
        f=(lambda x:x[0]**2+5*x[1]**3)
        g=(lambda x:[2*x[0],5*3*x[1]**2])
        Prob=Optimization.OptimizationProblem(2,f)
        Prob2=Optimization.OptimizationProblem(1,f,g)
        a=Prob.grad([100000,200000])
        b=Prob2.grad([100000,200000])
        for i,l in enumerate(a):
            self.assertTrue(abs(l-b[i])/abs(b[i])<0.01)
        a=Prob.grad([0.000001,0.000002])
        b=Prob2.grad([0.000001,0.000002])
        for i,l in enumerate(a):
            self.assertAlmostEqual(l,b[i])
    
    def test_Rosenbrock_classical_line(self):
        print("line")
        f=(lambda x:100*(x[1] - x[0]**2)**2 + (1-x[0])**2)
        g=(lambda x: np.array([400*x[0]**3-400*x[0]*x[1]+2*(x[0]-1), 200*(x[1]-x[0]**2)]))
        problem = Optimization.OptimizationProblem(2, f)
        solver = Optimization.ClassicalNewtonLine(problem, True)
        x0 = np.array([0.5, 0.5])
        x, f = solver.Find_Minimum(x0,1e-15,1e-15)
        x0_range = np.linspace(-0.5, 2, num=200)
        x1_range = np.linspace(-0.5, 4, num=200)
        solver.plot_solution(x0_range, x1_range)
        print(x, f)
        self.assertAlmostEqual(f, 0, 7)
        self.assertAlmostEqual(x[0], 1, 7)
        self.assertAlmostEqual(x[1], 1, 7)

    def test_Rosenbrock_classical_noline(self):
        print("noline")
        f=(lambda x:100*(x[1] - x[0]**2)**2 + (1-x[0])**2)
        g=(lambda x: np.array([400*x[0]**3-400*x[0]*x[1]+2*(x[0]-1), 200*(x[1]-x[0]**2)]))
        problem = Optimization.OptimizationProblem(2, f)
        solver = Optimization.ClassicalNewtonNoLine(problem, True)
        x0 = np.array([0.5, 0.5])
        x, f = solver.Find_Minimum(x0,1e-15,1e-15)
        x0_range = np.linspace(-0.5, 2, num=200)
        x1_range = np.linspace(-0.5, 4, num=200)
        solver.plot_solution(x0_range, x1_range)
        print(x, f)
        self.assertAlmostEqual(f, 0, 7)
        self.assertAlmostEqual(x[0], 1, 7)
        self.assertAlmostEqual(x[1], 1, 7)

    #Plotting of Rosenbrock function
    #def testingOptimizationPlot(self):
    #    f=(lambda x:100*(x[1] -x[0]**2)**2 + (1-x[0])**2)
    #    problem = Optimization.OptimizationProblem(2, f)
    #    x0_range = np.linspace(-0.5, 2, num =200)
    #    x1_range = np.linspace(-0.5, 4, num =200)
    #    problem.plot(problem, x0_range, x1_range, logspace = True)

    def test_Rosenbrock_BG(self):
        f=(lambda x:100*(x[1] - x[0]**2)**2 + (1-x[0])**2)
        g=(lambda x: np.array([400*x[0]**3-400*x[0]*x[1]+2*(x[0]-1),200*(x[1]-x[0]**2)]))
        problem = Optimization.OptimizationProblem(2, f)
        solver = Optimization.QuasiNewton(problem, True,'BG')
        x0=np.array([0,0])
        x,f=solver.Find_Minimum(x0,1e-15, 1e-8, True)
        x0_range = np.linspace(-0.5, 2, num=200)
        x1_range = np.linspace(-0.5, 4, num=200)
        #solver.plot_solution(x0_range,x1_range)
        #solver.plot_Hessapprox()
        print(x,f)
        #self.assertAlmostEqual(f,0,7)
        self.assertAlmostEqual(abs(x[0]-1),0,5)
        self.assertAlmostEqual(abs(x[1]-1),0,5)
    
    def test_Rosenbrock_BB(self):
        f=(lambda x:100*(x[1] - x[0]**2)**2 + (1-x[0])**2)
        g=(lambda x: np.array([400*x[0]**3-400*x[0]*x[1]+2*(x[0]-1), 200*(x[1]-x[0]**2)]))
        #f=(lambda x: x[1]**4+(x[0]-1)**4)
        #g=(lambda x: np.array([4*(x[0]-1),4*x[1]]))
        problem = Optimization.OptimizationProblem(2, f)
        solver = Optimization.QuasiNewton(problem, True,'BB')
        x0 = np.array([0,0])
        x, f = solver.Find_Minimum(x0, 1e-15, 1e-8, True)
        x0_range = np.linspace(-0.5, 2, num =200)
        x1_range = np.linspace(-0.5, 4, num =200)
        #solver.plot_solution(x0_range,x1_range)
        #solver.plot_Hessapprox()
        print(x, f)
        #self.assertAlmostEqual(f, 0, 7)
        self.assertAlmostEqual(abs(x[0]-1),0,5)
        self.assertAlmostEqual(abs(x[1]-1),0,5)

    def test_Rosenbrock_BFGS(self):
        f=(lambda x:100*(x[1] - x[0]**2)**2 + (1-x[0])**2)
        g=(lambda x: np.array([400*x[0]**3-400*x[0]*x[1]+2*(x[0]-1), 200*(x[1]-x[0]**2)]))
        problem = Optimization.OptimizationProblem(2, f)
        solver = Optimization.QuasiNewton(problem, False,'BFGS')
        x0 = np.array([2,2])
        x, f = solver.Find_Minimum(x0,1e-8,1e-15,True,True)
        x0_range = np.linspace(-0.5, 2, num=200)
        x1_range = np.linspace(-0.5, 4, num=200)
        solver.plot_solution(x0_range, x1_range)
        solver.plot_Hessapprox()
        print(x, f)
        self.assertAlmostEqual(f, 0)
        self.assertAlmostEqual(abs(x[0]-1),0,5)
        self.assertAlmostEqual(abs(x[1]-1),0,5)

    def testCheby(self):
        n_range = [4,8,11]
        for n in n_range:
            x0 = np.linspace(0,1,n)
            func = (lambda x: ChebyQuad_problem.chebyquad(x))
            func_grad = (lambda x: ChebyQuad_problem.gradchebyquad(x))
            problem = Optimization.OptimizationProblem(n, func, func_grad)
            solver = Optimization.QuasiNewton(problem, True, 'DFP')
            solution,f = solver.Find_Minimum(x0)
            correct_sol = so.fmin_bfgs(func, x0, func_grad, disp=False)
            print(correct_sol,func(correct_sol))
            print(solution,func(solution))
            self.assertAlmostEqual(np.linalg.norm(func_grad(solution)),0)

if __name__ == '__main__':
    unittest.main()