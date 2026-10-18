import numpy as np


class Dynamics():

    def __init__(self, dt=0.01, noisy=True):
        self.dt = dt
        self.noisy = noisy

    def feed_forward(self, state, control):
        raise NotImplementedError("Calling abstract function")

    def rollout(self, state_init, control_traj, num_rollouts):
        raise NotImplementedError("Calling abstract function")


class TurtleBotState():
    def __init__(self, x=0., y=0., th=0.):
        self.x = x
        self.y = y
        self.th = th


class TurtleBotControl():
    def __init__(self, v=0., om=0.):
        self.v = v
        self.om = om


class TurtleBotDynamics(Dynamics):

    def __init__(self, dt=0.01, noisy=True):
        super().__init__(dt, noisy)
        self.n = 3  # length of state
        self.m = 2  # length of control
        self.noise_std = np.array([0.01, 0.01, 0.001])

    def feed_forward(self, state: TurtleBotState, control: TurtleBotControl) -> TurtleBotState:
        """
        One Euler step of the unicycle model, plus Gaussian noise if self.noisy.
        Inputs:
            state: TurtleBotState, control: TurtleBotControl
        Output:
            New TurtleBotState
        """
        state_new = TurtleBotState()
        ########## Code starts here ##########
        state_new.x = state.x + control.v*np.cos(state.th)*self.dt
        state_new.y = state.y + control.v*np.sin(state.th)*self.dt
        state_new.th = state.th + control.om*self.dt
        ########## Code ends here ##########

        if self.noisy:
            w = np.random.normal(loc=np.zeros(self.n), scale=self.noise_std)
            state_new.x += w[0]
            state_new.y += w[1]
            state_new.th += w[2]
        return state_new

    def rollout(self, state_init: TurtleBotState, control_traj: np.ndarray, num_rollouts: int) -> np.ndarray:
        """
        Inputs:
            state_init: TurtleBotState
            control_traj: (m, num_steps) array, row 0 is v and row 1 is om
            num_rollouts: number of rollouts
        Output:
            Array of shape (num_rollouts, num_steps + 1, n), entry [i, k] is [x, y, th]
            of rollout i at step k (step 0 is state_init)

        HINT: use two for-loops and self.feed_forward.
        """
        num_steps = control_traj.shape[1]
        state_traj = np.zeros((num_rollouts, num_steps + 1, self.n))
        ########## Code starts here ##########
        state_traj[:,0,:] = np.tile(np.array([state_init.x,state_init.y,state_init.th]),(num_rollouts,1))
        for i in range(num_rollouts):
            for j in range(num_steps):
                # compute next state
                curr_control = TurtleBotControl(v=control_traj[0,j],om=control_traj[1,j])
                curr_state = TurtleBotState(x = state_traj[i,j,0],y = state_traj[i,j,1],th = state_traj[i,j,2])
                new_state = self.feed_forward(curr_state,curr_control)
                # save state
                state_traj[i,j+1,:] = np.array([new_state.x,new_state.y,new_state.th])

        ########## Code ends here ##########
        return state_traj


class DoubleIntegratorDynamics(Dynamics):

    def __init__(self, dt=0.01, noisy=True):
        super().__init__(dt, noisy)
        self.n = 4  # length of state
        self.m = 2  # length of control
        self.noise_std = np.array([0.01, 0.01, 0.001, 0.001])

        self.A = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.B = np.array([[0.0, 0.0],
                           [0.0, 0.0],
                           [self.dt, 0.0],
                           [0.0, self.dt]])

    def feed_forward(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        One Euler step for all rollouts at once, with no for-loops.
        Inputs:
            state: (n * num_rollouts,) stacked states [x, y, vx, vy] of every rollout
            control: (m,) control [ax, ay], the same for every rollout
        Output:
            (n * num_rollouts,) stacked next states
        """
        num_rollouts = state.shape[0] // self.n
        # Stack to step all rollouts in parallel
        A_stack = np.kron(np.eye(num_rollouts), self.A)
        B_stack = np.tile(self.B, (num_rollouts, 1))

        ########## Code starts here ##########
        state_new = A_stack @ state + B_stack @ control
    
        ########## Code ends here ##########

        if self.noisy:
            state_new = state_new + np.random.normal(loc=np.zeros(state.shape),
                                                     scale=np.tile(self.noise_std, num_rollouts))
        return state_new

    def rollout(self, state_init: np.ndarray, control_traj: np.ndarray, num_rollouts: int) -> np.ndarray:
        """
        Inputs:
            state_init: (n,) initial state, the same for every rollout
            control_traj: (m, num_steps) array
            num_rollouts: number of rollouts
        Output:
            Array of shape (num_rollouts, num_steps + 1, n), same layout as TurtleBotDynamics.rollout

        HINT: use only one for-loop over time, calling self.feed_forward once per step.
        """
        num_steps = control_traj.shape[1]
        state_traj = np.zeros((num_rollouts, num_steps + 1, self.n))
        ########## Code starts here ##########
        state_traj[:,0,:] = np.tile(state_init[None,:],(num_rollouts,1))

        for i in range(num_steps):
            Xi = state_traj[:,i,:].reshape(-1)
            Xn = self.feed_forward(Xi,control_traj[:,i])
            state_traj[:,i+1,:] = Xn.reshape(num_rollouts,self.n)
        ########## Code ends here ##########
        return state_traj
