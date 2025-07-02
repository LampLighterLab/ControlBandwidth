import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Actions, Gridworld
from value_estimator import MonteCarlo, TDLambda
from control_method import QLearning
import math

def plot_values(solver_obj):
    arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
    for i in range(solver_obj.env.SIZE_Y):
        for j in range(solver_obj.env.SIZE_X):
            arr[i,j] = solver_obj.value((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()

def test_monte_carlo():
    rewards = {
            (0,5):1
            }
    terminal_states = [
        (0,5)
        ]
    env = Gridworld(rewards, terminal_states, 10, 8)

    def up_policy(s):
        return Actions.UP

    mc = MonteCarlo(up_policy, (0,0), env, 0.9, 1000)

    start = time.time_ns()
    for i in range(n := 1):
        mc.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(mc)
    
def test_tdlambda():
    rewards = {
            (5,5):1
            }
    terminal_states = [
        (5,5)
        ]
    env = Gridworld(rewards, terminal_states, 10, 8, 5)

    def up_policy(s):
        return Actions.UP

    td = TDLambda(env.generate_random_action, (0,0), env, 0.9, 0.1, 0.5, 1000)

    start = time.time_ns()
    for i in range(n := 100):
        td.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(td)
    
def test_q_learning():
    rewards = {
            (7,8):1
            }
    terminal_states = [
        (7,8)
        ]
    env = Gridworld(rewards, terminal_states, 10, 10)
    def obs_func(s):
            x = s[0]-7
            y = s[1]-8
            return [x**2, y**2, 1]
    q = QLearning(env, (0,0), 0.2, 0.8, 0.5, obs_func)
    start = time.time_ns()
    for i in range(n := 500):
        q.episode()
    print(q.weights)
    q.optimal_path()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    
    solver_obj = q
    arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
    for i in range(solver_obj.env.SIZE_Y):
        for j in range(solver_obj.env.SIZE_X):
            arr[i,j] = solver_obj.state_value_approx((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()
    

    # Used AI to generate code for plot only
"""     actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Action-Value Function")

    for ax, action, name in zip(axs.flat, actions, action_names):
        arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
        for i in range(solver_obj.env.SIZE_Y):
            for j in range(solver_obj.env.SIZE_X):
                arr[i, j] = solver_obj.action_value(action, (j, i))

        im = ax.imshow(arr, origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_title(f"Action: {name}")

    plt.tight_layout()
    
    q.optimal_path()
    plt.show() """


#test_tdlambda()
#test_monte_carlo()
test_q_learning()

def generate_q_hyperparams():
    rewards = {
            (7,8):1
            }
    terminal_states = [
        (7,8)
        ]
    env = Gridworld(rewards, terminal_states, 10, 10)
    def obs_func(s):
            x = s[0]-7
            y = s[1]-8
            return [x, y, x**2, y**2, x*y, 1]

    import statistics
    hyperparam_space = np.zeros((9, 11))
    e = 1
    d = 0
    while (e <= 9):
        while (d <= 10):
            start = time.time_ns()
            for trials in range(5):
                q_solver = QLearning(env, (0,0), e/10, d/10, 0.5, obs_func)
                ratio = list()
                for episodes in range(500):
                    q_solver.episode()
                ratio.append(q_solver.state_value_approx((7,8)) / q_solver.state_value_approx((0,0)))
            hyperparam_space[e-1][d] = statistics.median(ratio)
            d += 1
            end = time.time_ns()
            print(d)
            print(e)
            print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
        e += 1
        d = 0
    
    print(hyperparam_space)

#generate_q_hyperparams()

def plot_q_hyperparams():
    # using obs vector [x^2, y^2, 1]
    hyperparam_space = np.array([
    [  -2.41482   , 20.81079   , -29.69531   , 128.57922   ,  -4.20834   ,  -1.41246   ,
      -1.21603   ,  -1.29426   , -11.77985   ,   6.3966    ,   1.        ],
    [  -3.01442   ,  -1.53253   , -15.20611   ,  -1.55717   ,  -1.85122   , 167.7134    ,
    -151.54563   ,  -3.95636   ,  -3.48186   ,  34.81809   ,   1.        ],
    [  -1.1998    , -10.31499   ,  -1.83877   ,  -1.14832   ,  -1.18648   ,  -8.38462   ,
      -1.74972   ,  -9.36039   ,  -2.21509   ,  13.32823   ,   1.        ],
    [  -1.72026   ,   7.27984   ,  -1.39925   ,  -2.25681   ,  -1.30944   ,  -2.02385   ,
     -12.04363   ,  -7.47626   , -10.42146   ,   3.38411   ,   1.        ],
    [  -1.44241   ,  -1.12715   ,  -2.10831   ,  -1.96234   ,  -1.61072   ,  -1.29369   ,
      -1.94765   ,  -5.40722   ,  -2.11048   ,  -7.60945   ,   1.        ],
    [  -1.39247   ,  -0.80145   ,  -0.90789   ,  -1.99657   ,  -2.18361   ,  -2.72005   ,
      -2.03773   ,  -1.7948    ,  -1.51859   ,  -7.32497   ,   1.        ],
    [  -1.76354   ,  -2.3288    ,  -1.78533   ,  -1.32754   ,  -0.9063    ,  -3.24532   ,
      -1.7742    , -32.08496   ,  -1.8432    ,  -2.31669   ,   1.        ],
    [  -0.79966   ,  -1.63033   ,  -1.24521   ,  -2.17248   ,  -4.9164    ,  -1.43564   ,
      -1.26788   ,  -0.85748   ,  -3.53858   ,  -3.28327   ,   1.        ],
    [  -6.15872   ,  -7.09923   ,  -1.74112   , -14.40401   ,  -1.4704    ,  -1.13041   ,
      -2.31638   ,  -1.01378   ,  -1.41869   ,  -1.76237   ,   1.        ]
    ])
    
    # using obs vector [x, y, x^2, y^2, xy, 1]
    '''hyperparam_space = np.array(
        [[   -3.12507 ,   62.43841  ,-174.84073 ,18767.13102 ,   -4.39889 ,   -1.10652,
     -4.74149  ,  -2.06048  , -75.94642  ,   5.6789  ,    0.99995],
 [   43.79638  ,   9.42516 , -133.94528  ,  -2.16849  ,  11.47432 , 3564.77817,
   4713.9858   ,  11.79562  ,  25.53245  ,  41.19207  ,   1.     ],
 [  -20.10007  ,   6.12953  ,  10.51278  ,   3.03494  ,   4.28118 ,  558.27452,
      3.97216  ,-178.59997  ,  17.40123  ,  14.15351  ,   1.     ],
 [    2.92032  , 201.72741 ,   -3.07525  ,   8.03679  ,  34.69949  ,  13.06222,
    -46.07467  , -16.24846 ,  -39.4147   ,   5.62648  ,   0.99994],
 [    3.36023  ,   2.24585  ,  -9.58686  ,   5.48952  ,  10.66742  ,   3.78712,
     -7.07662  , -10.87934 ,  -83.80517  , -51.15205  ,   0.99987],
 [   13.82672  ,  19.04703  ,   2.24419  ,  -4.88822  ,  16.17687  ,  31.4028,
      7.76619  ,   5.09752 ,    5.87849  ,  64.47205   ,  0.99995],
 [    6.91254   ,-30.76004  ,   5.71563  ,   2.46369   ,  5.70714  ,   4.45834,
     14.807    , 181.23092  ,  11.78195  ,  10.11377  ,   0.99993],
 [    2.36722  ,   3.92349  ,   4.22839  ,  79.14549  , -42.86587  ,   6.40321,
     12.00872 ,    3.41015  ,  11.83967  ,  18.09581  ,   1.     ],
 [   28.24229 , -141.52049  ,  32.69988  , -27.10098  ,   7.07983  ,   3.2841,
      9.37815  ,   6.53937   ,  4.38028  ,   4.53396  ,   0.99999]]
    )'''

    def symlog(x):
        if (x>1):
            return math.log10(x)
        elif (x>-1):
            return 0
        else:
            return -1 * math.log10(-1 * x)
    
    for i in range(9):
        for j in range(11):
            hyperparam_space[i][j] = symlog(hyperparam_space[i][j])

    plt.imshow(
        hyperparam_space,
        extent=(-0.05, 1.05, 0.05, 0.95),
        origin="lower"
    )
    

    plt.colorbar()
    plt.xlabel("discount factor")
    plt.ylabel("epsilon")
    plt.title("symlog(V(terminal state) / V(initial state)),\nmedian of 5 trials, 500 episodes,\nO(x) = [x^2,y^2,1]")
    plt.show()
    
#plot_q_hyperparams()