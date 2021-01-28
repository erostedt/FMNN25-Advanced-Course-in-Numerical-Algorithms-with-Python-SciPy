
import unittest
import numpy as np
import Solver
from Problem import RoomClass
from mpi4py import MPI


class Testing(unittest.TestCase):


    def test_solver(self):
        """
        Test of solve_reduced.
        """
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        bc = np.array([[1, 1]])
        b = np.array([3, 0, 1])     # Middle is unknown.
        x = Solver.solve_reduced(A, b, bc)
        sol = [-5/2, 1, 14/12]
        for i in range(3):
            self.assertAlmostEqual(sol[i], x[i])

    """def test_neumann(self):
        A = Solver.Solver.create_finite_diff(3)
        bc = np.array([[1, 1]])
        b = np.array([1, 2, 3])
        Anew, bnew = Solver.Solver.complementary_neumann_addition(A, b, bc)
        # A_true ="""


    def test_one_big_test(self):
        """

        :return:
        """
        init_guess = 15.
        normal_wall = 15.
        heater_wall = 40.
        window_wall = 5.
        width_x_r1r2r3 = 1
        length_y_r1r3 = 1
        length_y_r2 = 2
        boundary_cond_r1 = [(1, normal_wall), (2, init_guess), (3, normal_wall), (4, heater_wall)]
        boundary_cond_r2 = [(1, heater_wall), (2, init_guess), (3, normal_wall),
                            (4, window_wall), (5, init_guess), (6, normal_wall)]
        boundary_cond_r3 = [(1, normal_wall), (2, heater_wall), (3, normal_wall), (4, init_guess)]
        room1 = RoomClass(0, width_x_r1r2r3, length_y_r1r3, [2], boundary_cond_r1)
        room2 = RoomClass(1, width_x_r1r2r3, length_y_r2, [2, 5], boundary_cond_r2)
        room3 = RoomClass(2, width_x_r1r2r3, length_y_r1r3, [4], boundary_cond_r3)
        n = int(length_y_r1r3/room1.delta_x)
        m = int(length_y_r2/room2.delta_x)
        solver = Solver.Solver(n, m, [room1, room2, room3])
        solver.solve()
        temp_r1, temp_r2, temp_r3 = self.temp_room1, self.temp_room2, self.temp_room3
        solver.plot()
        print(temp_r1)


if __name__ == '__main__':
    unittest.main()
