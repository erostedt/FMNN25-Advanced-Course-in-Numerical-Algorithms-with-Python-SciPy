import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from Problem import RoomClass


def solve_reduced(A, b, bc):
    """
    Solves the reduced equation system. bc are all the Dirichlet conditions.
    :param A: Matrix A.
    :param b: Vector b.
    :param bc: Boundary conditions. Column 1 contains index and column 2 contains boundary value.
    :return: Solution to the reduced produced problem, x.
    """
    if len(bc) == 0:
        return np.linalg.solve(A, b)
    bc = np.array([i for i in bc if not (i[0] == 0 and i[1] == 0)])
    bc_indices = [int(i) for i in bc[:, 0]]
    bc_vals = bc[:, 1]
    n, _ = np.shape(A)
    x = np.zeros(n)
    non_bc = [i for i in range(n) if i not in bc_indices]
    sol = np.linalg.solve((A[non_bc, :][:, non_bc]), b[non_bc] - (A[non_bc, :][:, bc_indices] @ bc_vals))
    x[bc_indices] = bc_vals
    x[non_bc] = sol
    return x


class Solver:
    """
    Solver class. Indexation is done from top left and down along the columns.
    """

    def __init__(self, n, m, rooms, omega=0.8):
        """
        :param n: nodal points in y-direction
        :param m: nodal points in x-direction
        :param rooms: list of room-objects.
        :param omega: Parameter for slackness, optional, default = 0.8.
        """
        self.b = np.array([0] * n)
        self.rooms = rooms
        self.omega = omega
        self.n = n
        self.m = m
        self.A_middle = self.create_finite_diff(m**2//2, n, m)
        self.A1 = self.create_finite_diff(n**2, n, n)
        self.A2 = self.create_finite_diff(n**2, n, n)
        self.b1 = np.zeros(n**2)
        self.b2 = np.zeros(n**2)
        self.b = np.zeros(m**2//2)

    def solve(self):
        """
        Solves the heat problem by Dirichlet-Neumann method. Sets each rooms final temperature. 
        These temperatures can be accesed by self.temp_room1, self.temp_room2 and self.temp_room3.
        """
        comm = MPI.COMM_WORLD
        room_middle = self.rooms[1]
        room1 = self.rooms[0]
        room2 = self.rooms[2]
        shared_boundary_size1 = len(room_middle.boundaries[room_middle.shared_boundary[0]])
        shared_boundary_size2 = len(room_middle.boundaries[room_middle.shared_boundary[1]])
        starting_value1 = np.ones(shared_boundary_size1)*15
        starting_value2 = np.ones(shared_boundary_size2)*15
        leftproc = 1
        middleproc = 0
        rightproc = 2
        if comm.Get_rank() == leftproc:
            uk_right = np.zeros(n**2)
            comm.Send([starting_value1, MPI.FLOAT], dest=middleproc)
        if comm.Get_rank() == rightproc:
            uk_middle = np.zeros(n**2)
            comm.Send([starting_value2, MPI.FLOAT], dest=middleproc)
        if comm.Get_rank() == middleproc:
            uk_left = np.zeros(m**2//2)
        for iter in range(10):
            if comm.Get_rank() == middleproc:
                boundary1 = self._receive_mpi(shared_boundary_size1, 'float', leftproc, comm)
                boundary2 = self._receive_mpi(shared_boundary_size2, 'float', rightproc, comm)
                room_middle.boundaries[room_middle.shared_boundary[0], :, 1] = boundary1
                room_middle.boundaries[room_middle.shared_boundary[1], :, 1] = boundary2
                bc = np.concatenate(room_middle.boundaries)
                uk_middle1 = solve_reduced(self.A_middle, self.b, bc)     
                boundary1, boundary2 = self.set_derivatives(uk_middle1, self.m, dx=1/20)
                boundary1 = np.ascontiguousarray(boundary1[:, 1], dtype='float')
                boundary2 = np.ascontiguousarray(boundary2[:, 1], dtype='float') 
                comm.Send([boundary1, MPI.FLOAT], dest=leftproc) 
                comm.Send([boundary2, MPI.FLOAT], dest=rightproc)
                if iter>0:
                    uk_middle1 = self.omega*uk_middle1+(1-self.omega)*uk_middle
                uk_middle = uk_middle1
            if comm.Get_rank() == leftproc:
                bound = self._receive_mpi(shared_boundary_size1, 'float', middleproc, comm)
                room1.boundaries[room1.shared_boundary[0], :, 1] = bound
                self.A1, self.b1 = self.complementary_neumann_addition(self.A1, self.b1, room1.boundaries[room1.shared_boundary[0]], True)
                bc = np.copy(room1.boundaries)
                bc = np.delete(bc, room1.shared_boundary[0], axis=0)
                bc = np.concatenate(bc)
                uk_left1 = solve_reduced(self.A1, self.b1, bc)
                inds = [int(i) for i in room1.boundaries[room1.shared_boundary[0], :, 0]]
                if iter > 0:
                    uk_left1 = self.omega*uk_left1+(1-self.omega)*uk_left
                bounds = uk_left1[inds]
                if iter < 9:
                    comm.Send([bounds, MPI.FLOAT], dest=middleproc)
                uk_left = uk_left1
                
            if comm.Get_rank() == rightproc:
                bound = self._receive_mpi(shared_boundary_size2, 'float', middleproc, comm)
                room2.boundaries[room2.shared_boundary[0], :, 1] = bound
                self.A2, self.b2 = self.complementary_neumann_addition(self.A2, self.b2, room2.boundaries[room2.shared_boundary[0]], False)
                bc = np.copy(room2.boundaries)
                bc = np.delete(bc, room2.shared_boundary[0], axis=0)
                bc = np.concatenate(bc)
                uk_right1 = solve_reduced(self.A2, self.b2, bc)
                inds = [int(i) for i in room2.boundaries[room2.shared_boundary[0], :, 0]]
                if iter > 0:
                    uk_right1 = self.omega*uk_right1+(1-self.omega)*uk_right
                bounds = uk_right1[inds]
                if iter < 9:
                    comm.Send([bounds, MPI.FLOAT], dest=middleproc)
                uk_right = uk_right1
        if comm.Get_rank() == rightproc:
            comm.Send([uk_right, MPI.FLOAT], dest=middleproc)
        if comm.Get_rank() == leftproc:
            comm.Send([uk_left, MPI.FLOAT], dest = middleproc)
        if comm.Get_rank() == middleproc:
            uk_left = self._receive_mpi(self.n**2, 'float', leftproc, comm)
            uk_right = self._receive_mpi(self.n**2, 'float', rightproc, comm)
            self.temp_room1 = uk_left
            self.temp_room2 = uk_middle
            self.temp_room3 = uk_right

    @staticmethod
    def _receive_mpi(data_size, data_type, source, comm):
        """
        :param data_size: The size of the incoming array
        :param data_type: The data type that the incoming array consists of
        :param source: From which process the data is coming
        :param comm: The communicator
        :return: returns the received array of data
        """
        
        allocate_data = np.empty(data_size, dtype="float")
        comm.Recv([allocate_data, MPI.FLOAT], source=source)
        return allocate_data

    def complementary_neumann_addition(self, A, b, bc, right_bound):
        """
        Changes the A matrix and b vector to match with the neumann boundary conditions.
        :param A: Matrix A from Ax = b.
        :param b: Vector b from Ax = b.
        :param bc: Boundary conditions, n x 2 array with first entry being nodal index and second entry being nodal
        value.
        :param right_bound: Boolean value to check if it's the right boundary or the left.
        :return: Returns updated A and b.
        """
        if len(bc) == 0:
            return A, b

        m, _ = np.shape(A)
        dx_inv = (self.n - 1)
        if right_bound:
            start = m - 2 * self.n
            end = start + self.n
            for i in range(self.n):
                A[end + i, start + i] = dx_inv
                A[end + i, end + i] = -1*dx_inv
                b[end + i] = bc[i, 1]
        else:
            start = self.n
            end = 0
            for i in range(self.n):
                A[end + i, start + i] = -1*dx_inv
                A[end + i, end + i] = dx_inv
                b[end + i] = bc[i, 1]

        return A, b
            
    @staticmethod
    def create_finite_diff(size, n, m, dx=1/20):
        """
        :param size: The size of the finite difference matrix
        :param n: The number of y values in the mesh
        :param m: The number of x values in the mesh
        :param dx: The step size
        :return: The discretized laplace operator using a second order central difference
        """
        A = np.zeros((size, size))
        for i in range(1, n-1):
            for j in range(1, m-1):
                ind = m*i + j
                A[ind, ind+1] = dx**2
                A[ind, ind-1] = dx**2
                A[ind, ind] = -4*dx**2
                A[ind, ind+m] = dx**2
                A[ind, ind-m] = dx**2
        return A


    @staticmethod
    def set_derivatives(u, m, dx=1/20):
        """
        Set's the Neumann condition along the vertical lines.
        :param u: Temperature vector u.
        :param m: Dimension of matrix: m of mxm -> m
        :param dx: Step length dx.
        :return: Left and right Neumann boundary conditions.
        """
        room_1_start = m // 2
        room_3_start = m//2 * (m - 1)

        bc_left = np.zeros((room_1_start, 2))
        bc_right = np.zeros((room_1_start, 2))
        for i in range(1, room_1_start-1):
            bc_left[i, 0] = room_1_start + i
            bc_left[i, 1] = (u[room_1_start + i] - u[room_1_start + i + m]) / dx
            bc_right[i, 0] = room_3_start + i
            bc_right[i, 1] = (u[room_3_start + i] - u[room_3_start + i - m]) / dx
        bc_left[room_1_start-1, 0] = 2*room_1_start - 1
        bc_left[room_1_start-1, 0] = 0
        
        bc_left[0, 0], bc_left[0, 1] = room_1_start, (u[room_1_start] - u[room_1_start + m]) / dx
        bc_right[room_1_start-1, 0], bc_right[room_1_start-1, 1] = m - 1, (u[m - 1] - u[room_3_start - 1]) / dx
        return bc_left, bc_right

    def plot(self, show=True):
        """
        Method to plot the three rooms and their temperature distribution
        :param show: If the plot is to be shown directly, automatically set to true
        """
        xv_left, yv_left = self.rooms[0].xv, self.rooms[0].yv
        xv_middle, yv_middle = self.rooms[1].xv, self.rooms[1].yv
        xv_right, yv_right = self.rooms[2].xv, self.rooms[2].yv
        split_small = 1/(self.rooms[0].delta_x)
        split_large = 2*split_small
        temp_left = np.split(self.temp_room1, split_small)
        levels=np.linspace(5,40,20)
        temp_middle = np.flipud(np.reshape(self.temp_room2,(40,20),'F'))
        temp_right  = np.split(self.temp_room3, split_small)
        np.reshape(temp_middle ,(40,20))
        temp_left_flip = np.rot90(temp_left)
        temp_right_flip = np.rot90(temp_right)
        ay = np.empty([20,20])
        ay[:]=np.nan
        temp_left_nan = np.concatenate((temp_left_flip, ay))
        temp_right_nan = np.concatenate((ay,temp_right_flip))
        plt.subplot(131)
        plt.contour(xv_middle,yv_middle, temp_left_nan,levels=levels,vmax=40,vmin=5)
        plt.contourf(xv_middle,yv_middle,temp_left_nan,levels=levels,vmax=40,vmin=5)
        plt.axis('off')
        plt.subplot(132)
        plt.axis('off')
        plt.contour(xv_middle,yv_middle,temp_middle,levels=levels,vmax=40,vmin=5)
        plt.contourf(xv_middle,yv_middle,temp_middle,levels=levels,vmax=40,vmin=5)
        plt.subplot(133)
        plt.contour(xv_middle,yv_middle, temp_right_nan,levels=levels,vmax=40,vmin=5)
        plt.contourf(xv_middle,yv_middle,temp_right_nan,levels=levels,vmax=40,vmin=5)
        plt.axis('off')
        plt.colorbar()
        plt.subplots_adjust(wspace=0, hspace=0)
        if show:
            plt.show()


"""
Below follows the code that creates the three rooms, their respective boundary
conditions and then the solver that solves the problem. 
"""
init_guess = 15.
normal_wall = 15.
heater_wall = 40.
window_wall = 5.
width_x_r0r1r2 = 1
length_y_r0r2 = 1
length_y_r1 = 2
boundary_cond_r0 = [(1, normal_wall), (2, init_guess), (3, normal_wall), (4, heater_wall)]
boundary_cond_r1 = [(1, heater_wall), (2, init_guess), (3, normal_wall),
                    (4, window_wall), (5, init_guess), (6, normal_wall)]
boundary_cond_r2 = [(1, normal_wall), (2, heater_wall), (3, normal_wall), (4, init_guess)]
room0 = RoomClass(0, width_x_r0r1r2, length_y_r0r2, [2], boundary_cond_r0)
room1 = RoomClass(1, width_x_r0r1r2, length_y_r1, [2, 5], boundary_cond_r1)
room2 = RoomClass(2, width_x_r0r1r2, length_y_r0r2, [4], boundary_cond_r2)
n = int(length_y_r0r2/room0.delta_x)
m = int(length_y_r1/room1.delta_x)
solver = Solver(n, m, [room0, room1, room2])
solver.solve()

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    solver.plot()
