#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
import numpy as np
from numpy import linalg
import scipy
from scipy.interpolate import splev

import typing as T



class Navigator(BaseNavigator):
    def __init__(self, 
        kpx: float =1, 
        kpy: float =1, 
        kdx: float =1, 
        kdy: float =1,
        kpt: float =1) -> None:

        super().__init__("Navigator")

        self.V_PREV_THRES = 0.0001
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.kpt = kpt

        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

        # TODO: replace with actual one once I figure out where that is
        self.vel_heuristic = 0.15

        # TODO: remove test
        self.create_timer(1,self.test)
        self.cmd_pub = self.create_publisher(TurtleBotState, "/test", 10)

    def test(self):
        print("hello")
        state = TurtleBotState()
        state.x = 1.0
        state.y = 2.0
        state.theta = 3.0
        self.cmd_pub.publish(state)

    def compute_heading_control(
            self, state : TurtleBotState, 
            goal : TurtleBotState
    ) -> TurtleBotControl:
        err : float = wrap_angle(goal.theta-state.theta)
        omega : float = self.kpt * err
        res = TurtleBotControl()
        res.omega = omega
        return res

    def reset(self) -> None:
            self.V_prev = 0.
            self.om_prev = 0.
            self.t_prev = 0.


    def compute_trajectory_tracking_control(self, state : TurtleBotState, plan : TrajectoryPlan, t):
        """ 
        Compute control target using a trajectory tracking controller
        
        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        state_d = plan.desired_state(t)
        x_d = state_d.x
        y_d = state_d.y
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        dt = t - self.t_prev

        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        x = state.x
        y = state.y
        th = state.theta
        
        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                          [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        ctrl = TurtleBotControl()
        ctrl.omega = om
        ctrl.v = V
        return ctrl

    def compute_trajectory_plan(self,
            state: TurtleBotState,
            goal: TurtleBotState,
            occupancy: StochOccupancyGrid2D,
            resolution: float,
            horizon: float,
        ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """

        x_init = np.array([state.x,state.y]).astype(float)
        x_goal = np.array([goal.x,goal.y]).astype(float)

        # TODO: check if horizon is calculated this way
        # taking difference between goal and init states and padding up to horizon
        statespace_diff = np.abs(x_goal - x_init)
        statespace_max_size = np.array([horizon,horizon])
        statespace_padding = (statespace_max_size - statespace_diff)/2
        if np.any(statespace_padding < 0):
            raise ValueError("Calculated State Space Padding is negative")

        # update relevant properties
        self.occupancy = occupancy
        self.goal = goal

        statespace_lo = np.array([
            min(x_init[0],x_goal[0]), 
            min(x_init[1],x_goal[1])
            ]) - statespace_padding
        statespace_hi = np.array([
            max(x_init[0],x_goal[0]), 
            max(x_init[1],x_goal[1])
            ]) + statespace_padding

        astar = AStar(statespace_lo,statespace_hi,x_init,x_goal,occupancy,float(resolution))
        print("test")
        if not astar.solve():
            return None

        elif len(astar.path) < 4:
            return None
        


        # reset class variables for previous velocity and time
        self.reset()

        # smooth trajectory plan
        smooth_plan : TrajectoryPlan = self.compute_smooth_plan(astar.path, self.vel_heuristic, 0.05)

        return smooth_plan
        # TODO: check algorithm
        # QUESTIONS: Where to get velocity target to compute time stamps?
        #            Is my understanding of "horizon" arg correct?
        #            Recommendations for debugging? Just kind of one shotting this

    def compute_smooth_plan(self, path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        # Ensure path is a numpy array
        path = np.asarray(path)
        # Compute and set the following variables:
        #   1. ts:
        #      Compute an array of time stamps for each planned waypoint assuming some constant
        #      velocity between waypoints.
        #
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.
        #      Hint: Use scipy.interpolate.splrep
        ##### YOUR CODE STARTS HERE #####
        
        distances = np.array([
            np.linalg.norm(np.array(next_node)-np.array(curr_node)) 
            for curr_node, next_node in zip(path[:-1],path[1:])
        ])
        distances = np.concatenate(([0],distances))
        
        tsteps = distances / v_desired
        ts = np.cumsum(tsteps)
        path_x_spline = scipy.interpolate.splrep(ts,path[:,0],s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts,path[:,1],s=spline_alpha)
        ###### YOUR CODE END HERE ######

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

        
            

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""
    occupancy :StochOccupancyGrid2D
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1,
                 heuristic="l2", heuristic_weight=1.0):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.heuristic_type = heuristic            # norm used by the heuristic: "l1", "l2", or "linf"
        self.heuristic_weight = heuristic_weight   # weight w >= 1 on the heuristic (w > 1 gives weighted A*)
        self.x_offset = x_init             
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state


        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.heuristic(self.x_init)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        
        ########## Code starts here ##########
        states_in_bounds = [xi >= xi_lo and xi <= xi_hi for xi, xi_lo, xi_hi in zip(x,self.statespace_lo,self.statespace_hi)]
        return all(states_in_bounds) and self.occupancy.is_free(x) 
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states. This is the cost of
        traveling along the edge between two neighboring states, and it is used
        for the cost-to-arrive. It should NOT depend on self.heuristic_type.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.sqrt(np.sum(np.square(np.array(x1) - np.array(x2))))
        ########## Code ends here ##########

    def heuristic(self, x):
        """
        Computes the weighted heuristic estimate of the cost-to-go from state x
        to self.x_goal, i.e. w * ||x - x_goal||, where w = self.heuristic_weight
        and the norm is selected by self.heuristic_type:
            "l1":   sum of absolute coordinate differences
            "l2":   Euclidean norm
            "linf": largest absolute coordinate difference
        Input:
            x: state tuple
        Output:
            Float heuristic value

        HINT: np.linalg.norm accepts an `ord` argument. Raise a ValueError for
              an unrecognized self.heuristic_type.
        """
        ########## Code starts here ##########
        w = self.heuristic_weight
        xdiff = np.array(self.x_goal) - np.array(x)
        h : float = np.inf
        if self.heuristic_type == "l1":
            h = np.sum(np.abs(xdiff))
        elif self.heuristic_type == "l2":
            h = np.linalg.norm(xdiff)
        elif self.heuristic_type == "linf":
            h = np.max(np.abs(xdiff))
        else:
            raise RuntimeError("Unexpected state, invalid heuristic")
        
        return h*w
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should
                make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        #writing code that extends to N-d state space
        dx = self.resolution
        N = len(x)
        #each direction has 3 states, jog backward, no jog, jog forward
        jog_amounts = (-dx,0,dx)
        #breakpoint()
        def generate_neighbors(jog_list):
            if len(jog_list) == N:
                #base case
                neighbor = self.snap_to_grid(np.array(x)+np.array(jog_list))
    
                if np.sum(np.abs(np.array(jog_list))) == 0:
                #avoid adding itself as a neighbor
                    return
                elif self.is_free(np.array(neighbor)):
                    neighbors.append(neighbor)
            else:
            # recursive case
                for jog in jog_amounts:
                    generate_neighbors(jog_list + (jog,))
        
        empty = ()
        generate_neighbors(empty)
        ########## Code ends here ##########
        
        
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    # def plot_path(self, fig_num=0, show_init_label=True):
    #     """Plots the path found in self.path and the obstacles"""
    #     if not self.path:
    #         return

    #     self.occupancy.plot(fig_num)

    #     solution_path = np.asarray(self.path)
    #     plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
    #     plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
    #     if show_init_label:
    #         plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
    #     plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

    #     plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    # def plot_tree(self, point_size=15):
    #     plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
    #     plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
    #     px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
    #     py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
    #     plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".

                Use self.distance for edge costs (cost-to-arrive) and
                self.heuristic for the estimated cost-to-go, as in Algorithm 1
                of the handout.
        """
        ########## Code starts here ##########
#         initialize list to explore by starting at our inital position
        # turns out that was done in constructor
        while len(self.open_set)>0:
#             pop the lowest cost through off the list to explore
            curr_x = self.find_best_est_cost_through()
            self.open_set.remove(curr_x)
            self.closed_set.add(curr_x)
            
            # if the state we are exploring is the target state, run path reconstruction
            if self.distance(curr_x, self.x_goal) < self.resolution/10:
#                 give a margin just in case any weird rounding thing happens
                self.path = self.reconstruct_path()
                return True

            neighbors = self.get_neighbors(curr_x)
            # for each neighbor:
            for neighbor in neighbors:
                # do nothing if it is already in the closed set
                if neighbor in self.closed_set:
                    continue
                
                # ecalculate the total cost to get to the neighbor
                tentative_cost_to_arrive = self.cost_to_arrive[curr_x] + self.distance(curr_x,neighbor)
                # if the neighbor is not in the open set, add it
                if not neighbor in self.open_set:
                    self.open_set.add(neighbor)
                
                elif tentative_cost_to_arrive > self.cost_to_arrive[neighbor]:
                    # else, if the cost to get to it is higher than the one it has, dont do anything to it
                    continue
                    
                # update dictionaries
                self.came_from[neighbor] = curr_x
                self.cost_to_arrive[neighbor] = tentative_cost_to_arrive
                self.est_cost_through[neighbor] = tentative_cost_to_arrive + self.heuristic(neighbor)
        return False
        ########## Code ends here ##########


def main():
    rclpy.init()
    nav = Navigator()
    rclpy.spin(nav)
    rclpy.shutdown()


if __name__ == "__main__":
    main()