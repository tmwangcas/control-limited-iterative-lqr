import numpy as np
# import autograd.numpy as np
# from autograd import grad, jacobian

from iLQR import iLQR
import rendering
import time
from os import path


"""A demo of iLQG/DDP with inverted pendulum swing-up dynamics"""

class InvertedPendulumEnv:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # env parameters
        self.m  = 1
        self.l  = 1
        self.g  = 10
        self.dt = 0.05
        self.T  = 200

        # cost function parameters
        # self.Q1 = 1
        # self.Q2 = 0.1
        # self.R  = 0.001
        self.Q1 = 10
        self.Q2 = 0.1
        self.R  = 0.001

        # state and action
        self.x      = None
        self.u      = None
        self.x_dim  = 3
        self.u_dim  = 1
        self.u_lims = np.array([[-2, 2]])  # control limits

        # render parameters
        self.viewer         = None
        self.pole_transform = None
        self.img            = None
        self.imgtrans       = None

    # dynamic model
    def dynamics(self, x, u):  # 2d array: x dim = [n, N+1], u dim = [m, N+1]
        # theta = np.arctan2(x[0], x[1])
        # theta_new = theta + self.dt * x[2]
        #
        # xdd = 3 * self.g / (2 * self.l) * x[0] + 3 / (self.m * self.l ** 2) * u[0]  # 1d array
        # omega_new = x[2] + self.dt * xdd
        #
        # x_new = np.array([np.sin(theta_new), np.cos(theta_new), omega_new])

        x_new0 = x[0] * np.cos(x[2]*self.dt) + x[1] * np.sin(x[2]*self.dt)
        x_new1 = x[1] * np.cos(x[2]*self.dt) - x[0] * np.sin(x[2]*self.dt)
        x_new2 = x[2] + 3*self.g/(2*self.l)*x[0]*self.dt + 3/(self.m*self.l**2)*u[0]*self.dt

        x_new = np.array([x_new0, x_new1, x_new2])

        return x_new  # 2d array

    def dynamics_dx(self, x, u):
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        dfdx = np.zeros((N,n,n))
        for i in range(N):
            dfdx[i,:,:] = np.array([
                [ np.cos(x[2,i]*self.dt), np.sin(x[2,i]*self.dt), (-x[0,i]*np.sin(x[2,i]*self.dt) + x[1,i]*np.cos(x[2,i]*self.dt)) * self.dt],
                [-np.sin(x[2,i]*self.dt), np.cos(x[2,i]*self.dt), (-x[0,i]*np.cos(x[2,i]*self.dt) - x[1,i]*np.sin(x[2,i]*self.dt)) * self.dt],
                [3*self.g/(2*self.l) * self.dt, 0, 1]
            ])
        return dfdx  # 3d array

    def dynamics_du(self, x, u):
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        dfdu = np.zeros((N,n,m))
        for i in range(N):
            dfdu[i,:,:] = np.array([
                [0],
                [0],
                [3 * self.dt / (self.m * self.l ** 2)]
            ])
        return dfdu  # 3d array

    # cost function
    def cost(self, x, u):
        return x[0] ** 2 * self.Q1 + (x[1] - 1) ** 2 * self.Q1 + x[2] ** 2 * self.Q2 + u[0] ** 2 * self.R  # 1d array (cost is scalar)

    def cost_dx(self, x, u):
        dldx = np.array([2 * x[0] * self.Q1,
                         2 * (x[1] - 1) * self.Q1,
                         2 * x[2] * self.Q2])
        return dldx  # 2d array

    def cost_du(self, x, u):
        dldu = np.array([2 * u[0] * self.R])
        return dldu  # 2d array

    def cost_dxx(self, x, u):
        n = x.shape[0]
        N = x.shape[1]
        dldxx = np.zeros((N,n,n))
        for i in range(N):
            dldxx[i,:,:] = np.array([
                [2 * self.Q1, 0, 0],
                [0, 2 * self.Q1, 0],
                [0, 0, 2 * self.Q2]
            ])
        return dldxx  # 3d array

    def cost_duu(self, x, u):
        m = u.shape[0]
        N = u.shape[1]
        dlduu = np.zeros((N,m,m))
        for i in range(N):
            dlduu[i,:,:] = np.array([
                [2 * self.R]
            ])
        return dlduu  # 3d array

    def cost_dux(self, x, u):
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        dldux = np.zeros((N,m,n))
        for i in range(N):
            dldux[i,:,:] = np.array([
                [0, 0, 0]
            ])
        return dldux  # 3d array

    # combine dynamics and cost
    def dyn_cst(self, x, u):
        # n = x.shape[0]
        # m = u.shape[0]
        # N = x.shape[1]

        # dynamics and cost
        f = self.dynamics(x, u)
        c = self.cost(x, u)

        # dynamics first derivatives
        fx = self.dynamics_dx(x, u)
        fu = self.dynamics_du(x, u)

        # dynamics_dx = jacobian(self.dynamics, 0)
        # dynamics_du = jacobian(self.dynamics, 1)
        # fx = np.zeros((N,n,n))
        # fu = np.zeros((N,n,m))
        # for i in range(N):
        #     fx[i,:,:] = dynamics_dx(x[:,i], u[:,i])
        #     fu[i,:,:] = dynamics_du(x[:,i], u[:,i])
        # print(fx)
        # print(fx.shape)

        # cost first derivatives
        cx = self.cost_dx(x, u)
        cu = self.cost_du(x, u)

        # cost second derivatives
        cxx = self.cost_dxx(x, u)
        cuu = self.cost_duu(x, u)
        cux = self.cost_dux(x, u)

        return f, c, fx, fu, cx, cu, cxx, cuu, cux

    def perform_trajectory(self, x, u):
        total_reward = 0
        for i in range(self.T):
            self.x = x[:,i]  # 1d array
            self.u = u[:,i]  # 1d array
            self.render()
            x_reduced = self.reduce_state(self.x)
            total_reward += x_reduced[0]**2 + 0.1*x_reduced[1]**2 + 0.001*self.u[0]**2
        return total_reward

    def reset(self):
        # high = np.array([np.pi, 1])
        high = np.array([[np.pi], [1]])
        x_reduced = np.random.uniform(low=-high, high=high)
        self.x = self.augment_state(x_reduced)
        return self.x

    def step(self, u):
        self.u = u  # for rendering
        costs = self.cost(self.x, u)
        self.x = self.dynamics(self.x, u)
        return self.x, -costs, False

    def set_state(self, x):
        self.x = x

    def get_state(self):
        return self.x

    def _get_obs(self):
        return np.array([np.cos(self.x[0]), np.sin(self.x[0]), self.x[1]])

    @staticmethod
    def augment_state(x):
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])

    @staticmethod
    def reduce_state(x):
        theta = np.arctan2(x[0], x[1])
        return np.array([theta, x[2]])

    @staticmethod
    def angle_normalize(theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def render(self, mode='human'):
        if self.viewer is None:
            # from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        theta = np.arctan2(self.x[0], self.x[1])
        self.pole_transform.set_rotation(theta + np.pi/2)
        if self.u[0]:
            self.imgtrans.scale = (-self.u[0]/2, np.abs(self.u[0])/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


if __name__ == '__main__':

    print('\nA demonstration of the iLQG algorithm with car parking dynamics.\n'
          'for details see\n'
          'Tassa, Mansard & Todorov, ICRA 2014\n'
          '\"Control-Limited Differential Dynamic Programming\"\n')

    # create env
    env = InvertedPendulumEnv()

    # set random seed
    # np.random.seed(0)

    # set up the optimization problem
    # theta0     = np.pi  # initial angle
    # x0         = np.array([[np.sin(theta0)], [np.cos(theta0)], [0]])  # initial state
    x0         = env.reset()  # reset state
    x0_reduced = env.reduce_state(x0)

    # u0     = np.tile(np.array([[0]]), env.T)  # zero initial controls
    u_high = np.tile(np.array([[0.01]]), env.T)
    u0     = np.random.uniform(low=-u_high, high=u_high)  # random initial controls

    DYNCST = lambda x, u: env.dyn_cst(x, u)

    # === run the optimization
    x, u, cost = iLQR(DYNCST, x0, u0, env.u_lims)
    x_reduced = env.reduce_state(x)

    # === perform optimized trajectory
    for i in range(3):
        total_reward = env.perform_trajectory(x, u)
        print('Initial State: [%6.3fpi, %6.3f] | Reward: %9.3f' % (x0_reduced[0,0]/np.pi, x0_reduced[1,0], total_reward))
        time.sleep(1)
    env.close()

