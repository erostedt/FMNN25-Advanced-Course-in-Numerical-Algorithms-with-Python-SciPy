from bisect import bisect
import matplotlib.pyplot as plt
import unittest
import numpy as np
import scipy.linalg as la


class Spline:

    def __init__(self, u_interval, d_interval=[],padding = True):
        """ 
        init function, sets private attributes

        Parameters
        ----------
        u_interval : list of floats or integers
            Corresponds to the u interval [u2, uk+2]
        d_interval : list of tuples of floats or integers
            Corresponds to the control points of the spline,
            can be left empty for when class is used for interpolation purposes
        padding : boolean
            Automatically set as true, if set as false padding of u interval must be done
            before calling class init method
            
        The u_interval is padded with the boundary conditions u_0=u_1=u_2, u_(K-2)=u_(K-1)=u_K
        """
        u_interval.sort()
        self.u_interval = u_interval
        if padding:
            frontpad=self.u_interval[0]
            endpad=self.u_interval[-1]
            self.u_interval = [frontpad]*2+self.u_interval+[endpad]*2
        self.d_interval = d_interval

    def __call__(self, u):
        """ 
        Class call function, executes when instance of class is called

        This method will use the Blossom algorithm to compute the spline value
        given the value u, a u-interval given in the init method and a d-interval
        also given in the init method.

        Parameters
        ----------
        u : integer or float
            the value in which the spline is to be evaluated
        
        Returns
        -------
        list[float]
            Returns the x and y value of s(u) in a list of floats
        """
        self.u = u
        index = bisect(self.u_interval, self.u) - 1
        if index==len(self.u_interval)-1: #In order to evaluate the last point(bisect puts us to the right of the point)
            index=len(self.u_interval)-4
        u_interval_small = self.u_interval[index - 2: index + 4]
        d_interval_small = self.d_interval[index - 2: index + 2]
        s_u=self._blossom(d_interval_small, u_interval_small)
        return s_u
    
    def setd(self,d):
        """
        Method to add a d interval to instance of class

        Parameters
        ----------
        d : list[tuple]
            a list of the control points in d-interval, where each point is a tuple (x,y)
        """
        self.d_interval=d
    
    def _alpha(self, u_right, u_left):
        """
        Evaluates alpha according to the formula. 

        Parameters
        ----------
        u_right : int
            The "most right knot" in the u interval used for calculating a particular blossom.
        u_left : int
            The "most left knot" in the u interval used for calculating a particular blossom. 

        Returns 
        -------
        int
            Returns the value of the alpha constant 
        """
        return (u_right - self.u)/(u_right-u_left)

    def _blossom(self, d_interval_small, u_interval_small):
        """
        Evaluates the spline and the blossoms according to the algorithm and then 
        returns the value for a spline for a particular u value. 

        Parameters
        ----------
        d_interval_small : list with tuples
            List with the d values requiered for the blossom algortihm. 
        u_interval_small : u_interval_small :  list with floats or ints
            List with the u values requiered for evaluating the diffrent blossoms. 

        Returns 
        -------
        int
            The spline value for a particualar u-value.
        """
        blossom = []
        index = 2
        for i in range(3):
            alpha = self._alpha(u_interval_small[index+i+1], u_interval_small[index+i-2])
            tuple1 = [alpha * j for j in d_interval_small[i]]
            tuple2 = [(1 - alpha) * j for j in d_interval_small[i + 1]]
            d_step = tuple([tuple1[j] + tuple2[j] for j in range(len(tuple1))])
            blossom.append(d_step)
        blossom2 = []
        for i in range(2):
            alpha = self._alpha(u_interval_small[index+i+1], u_interval_small[index+i-1])
            tuple1 = [alpha * j for j in blossom[i]]
            tuple2 = [(1 - alpha) * j for j in blossom[i + 1]]
            d_step = tuple([tuple1[j] + tuple2[j] for j in range(len(tuple1))])
            blossom2.append(d_step)
        alpha = self._alpha(u_interval_small[index+1], u_interval_small[index])
        tuple1 = [alpha * j for j in blossom2[0]]
        tuple2 = [(1 - alpha) * j for j in blossom2[1]]
        s_u = [tuple1[j] + tuple2[j] for j in range(len(tuple1))]
        self.b0, self.b1, self.b2 = d_interval_small[0],blossom[0],blossom2[0]
        return s_u

    def plot(self, controlpoints=False, ind=-1):
        """
        Plot method of the class, to plot the spline

        Parameters
        ----------
        controlpoints : boolean
            If true, controlpoints will be plotted with spline. Otherwise set to false.
        ind : int
            Set to -1 by default, which means that blossom will not be plotted,
            if other value is added the blossoms will be plotted around this index
        """
        us = np.linspace(self.u_interval[2], self.u_interval[-2],1000)
        x = [0] * len(us)
        y = [0] * len(us)
        xBlossom=[]
        yBlossom=[]
        for i in range(len(us)):
            val = self(us[i])
            x[i] = val[0]
            y[i] = val[1]
            #if ind is specified, the blossoms are stored to be plotted
            if ind>-1 and us[i]>self.u_interval[ind] and us[i]<self.u_interval[ind+1]:
                xBlossom.extend([self.b1[0], self.b2[0]])
                yBlossom.extend([self.b1[1], self.b2[1]])
        plt.plot(x, y)
        if(controlpoints):
            dx=[i[0] for i in self.d_interval]
            dy=[i[1] for i in self.d_interval]
            plt.plot(dx,dy,'--ro',zorder=1)
        plt.scatter(xBlossom, yBlossom, color='green', zorder=2)
        if ind>-1:
            plt.scatter(self.d_interval[ind][0],self.d_interval[ind][1],s=120,color='magenta', zorder=2)
        plt.show()

    def N(self, i, k, u):
        """
        Calculates basis function N.
        :param i: Index i.
        :param k: Order, in this version it only works for k = 3 (cubic spline).
        :param u: Scalar u.
        :return: Basis function N.
        """
        if u == self.u_interval[len(self.u_interval) - 1] and i == len(self.u_interval) - 3:
            return 1
        j = bisect(self.u_interval, u)
        if k == 0:
            if i > 0 and self.u_interval[i - 1] == self.u_interval[i]:
                return 0
            return 1 if i == j else 0
        if (i - 1 < 0 or (self.u_interval[i + k - 1] - self.u_interval[i - 1]) == 0) and (
                i + k > len(self.u_interval) - 1 or (
                self.u_interval[i + k] - self.u_interval[i]) == 0):
            return 0
        elif i - 1 < 0 or (self.u_interval[i + k - 1] - self.u_interval[i - 1]) == 0:
            return (self.u_interval[i + k] - u) / (self.u_interval[i + k] - self.u_interval[i]) * self.N(i + 1, k - 1,u)
        elif i + k > len(self.u_interval) - 1 or (self.u_interval[i + k] - self.u_interval[i]) == 0:
            return (u - self.u_interval[i - 1]) / (self.u_interval[i + k - 1] - self.u_interval[i - 1]) * self.N(i,k - 1,u)
        else:
            return (u - self.u_interval[i - 1]) / (self.u_interval[i + k - 1] - self.u_interval[i - 1]) * self.N(i,k - 1,u) + (
                           self.u_interval[i + k] - u) / (self.u_interval[i + k] - self.u_interval[i]) * self.N(
                i + 1, k - 1, u)

    def greville(self):
        """
        Calculates Greville Abscissae points.
        :return: Vector with Greville Abscissae points xi_i.
        """
        return [(self.u_interval[i] + self.u_interval[i + 1] + self.u_interval[i + 2]) / 3 for i in
                range(len(self.u_interval) - 2)]

    def solve_Vandermonde(self, x, y):
        """
        Solves equation systems:
        N(xi)d_x = x
        N(xi)d_y = y
        :param x: vector with x values.
        :param y: vector with y values.
        :return: (d_x, d_y)
        """
        K = len(self.u_interval) - 1
        L = K - 2
        xis = self.greville()
        vandermonde = np.zeros((L + 1, L + 1))
        for r in range(L + 1):
            for c in range(L + 1):
                vandermonde[r, c] = self.N(c, 3, xis[r])
        return list(zip(la.solve(vandermonde, x), la.solve(vandermonde, y)))


class Test(unittest.TestCase):

    def test_N0(self):
        """
        Tests if N_0 works according to the definition
        """
        u_interval = [1, 3, 4, 6]
        spline=Spline(u_interval)
        for i in [1,2,4,5,6]:
            self.assertEqual(spline.N(i,0,1.5),0)
        self.assertEqual(spline.N(3,0,1.5),1)

    def test_div_by_zero(self):
        """
        Tests if the basis function returns 0, if two nodes coincide, according to the definition of 0/0
        """
        u_interval = [1, 2, 2, 3, 6]
        spline = Spline(u_interval)
        self.assertEqual(spline.N(1, 1, 2), 0)
    
    def test_N3(self):
        """
        The cubic spline basis functions are computed both recursively, and with the blossom algorithm, and compared.
        """
        u_interval = [1, 2, 2, 2.5, 3, 6]
        for i in range(7):
            d=[(0,0)]*i+[(1,0)]+[(0,0)]*(7-i)
            spline=Spline(u_interval,d)
            for j in np.linspace(1,6,50):
                self.assertAlmostEqual(spline.N(i,3,j),spline(j)[0])
    
    def test_if_s_eq_sum(self):
        """
        s(u) is computed for u=5.99 using blossom algorithm, and recursively generated basis functions, should be equal.
        """
        u_interval = [1, 3, 4, 5, 6, 7]
        d_interval = [(1, 2), (2, 3), (3,2), (3, 4), (3.2,3), (4, 4), (3,6), (4.5,4.5)]
        K = len(u_interval) - 1 + 4 #+4 for padding
        u = 5.99
        spline = Spline(u_interval, d_interval)
        s_u = spline(u)
        spline.plot(True, 3)
        sum_0, sum_1 = 0, 0
        for i in range(0, K - 2):
            sum_0 += d_interval[i][0] * spline.N(i, 3, u)
            sum_1 += d_interval[i][1] * spline.N(i, 3, u)
        self.assertAlmostEqual(sum_0, s_u[0])
        self.assertAlmostEqual(sum_1, s_u[1])
    def test_interp(self):
        """
        Test if interpolation points lie on spline.
        """
        u_interval = [1, 3, 4, 5, 6, 7] #knot intervals
        #Interpolation points
        x = [1, 2, 3, 3, 3.2, 4, 5, 4] 
        y = [2, 3, 2, 4, 3, 4, 1, 2]
        spline = Spline(u_interval)
        ds = spline.solve_Vandermonde(x, y)
        spline.setd(ds)
        xi=spline.greville()#The greville points s(xi)=(xi,yi)
        spline.plot(True)
        for i,j in enumerate(xi):
            self.assertAlmostEqual(spline(j)[0],x[i])
            self.assertAlmostEqual(spline(j)[1],y[i])
    def test_greville(self):
        """
        Testing MA computation with manually calculated.
        """
        u_interval = [1, 2]
        spline = Spline(u_interval, [])
        test_v = [1, (1+1+2)/3, (1+2+2)/3, (2+2+2)/3]
        greville=spline.greville()
        for j in range(4):
            self.assertAlmostEqual(test_v[j],greville[j])

    def test_claus_points(self):
        """
        Testing the points given from Claus in Canvas
        """
        d_interval = [(-12.73564, 9.03455),
                    (-26.77725, 15.89208),
                    (-42.12487, 20.57261),
                    (-15.34799, 4.57169),
                    (-31.72987, 6.85753),
                    (-49.14568, 6.85754),
                    (-38.09753, -1e-05),
                    (-67.92234, -11.10268),
                    (-89.47453, -33.30804),
                    (-21.44344, -22.31416),
                    (-32.16513, -53.33632),
                    (-32.16511, -93.06657),
                    (-2e-05, -39.83887),
                    (10.72167, -70.86103),
                    (32.16511, -93.06658),
                    (21.55219, -22.31397),
                    (51.377, -33.47106),
                    (89.47453, -33.47131),
                    (15.89191, 0.00025),
                    (30.9676, 1.95954),
                    (45.22709, 5.87789),
                    (14.36797, 3.91883),
                    (27.59321, 9.68786),
                    (39.67575, 17.30712)]
        KNOTS = np.linspace(0, 1, 26)
        KNOTS[ 1] = KNOTS[ 2] = KNOTS[ 0]
        KNOTS[-3] = KNOTS[-2] = KNOTS[-1]
        spline = Spline(list(KNOTS), d_interval, False)
        self.assertAlmostEqual(spline(0.2)[0],-31.90219167) 
        self.assertAlmostEqual(spline(0.2)[1],6.47655833)
        spline.plot(True, 3)


    
if __name__ == '__main__':
    unittest.main()

