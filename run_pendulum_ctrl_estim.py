import matplotlib.pyplot as plt
import numpy as np
import time
from value_estimator import MonteCarloDiscrete, TDLambdaDiscrete, MonteCarloContinuous
from inverted_pendulum import InvertedPendulum
from environment import Actions, Gridworld
from control_method import QLearning, ReinforceSoftmax, QLearningPendulum

# Create pendulum environment and instance of QLearningPendulum
# 0 < control_timestep <= sim_timestep
def initialize_env_and_solver(sim_timestep, control_timestep):
    params = {"mass":1, "length":1, "gravity":1}
    initial_state = np.array([0, 0.1])
    env = InvertedPendulum(params=params, initial_state=initial_state,
                           sim_timestep = sim_timestep, control_timestep= control_timestep)
    q_solver = QLearningPendulum(env, initial_state=initial_state, epsilon=0,
                                 discount_factor=0, learning_rate=1e-5, max_timestep=20)
    return q_solver

# Run episodes of Q-learning control method, return weights of state-value-function approximation
# num_episodes > 0
def get_approx_weights(q_solver, num_episodes):
    start = time.time_ns()
    print("Running Q-Learning value function approximation")
    for i in range(n := num_episodes):
        prev_weight_vector = q_solver.weights
        q_solver.run_episode()
        weight_update_size = np.linalg.norm(np.subtract(q_solver.weights, prev_weight_vector))
        print(f"Running episode {q_solver.episode_count}, magnitude of weight update is {weight_update_size}")
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    return q_solver.weights

# Run episodes of Monte Carlo value function estimation, return weights of state-value-function approximation
def get_true_weights(q_solver):
    random_generator = np.random.default_rng(123)
    params = {"mass":1, "length":1, "gravity":1}
    initial_state = np.array([0.1, 0])
    env = InvertedPendulum(params=params, initial_state=initial_state,
                           sim_timestep=q_solver.env.sim_timestep, control_timestep=q_solver.env.control_timestep)
    mc = MonteCarloContinuous(initial_state, env, discount_factor=0, policy=q_solver.get_next_action, max_timestep=q_solver.max_timestep)
    
    #Run episodes
    start = time.time_ns()
    print("Running Monte Carlo value function estimation")
    for i in range(n := 100):
        prev_weight_vector = mc.weights
        mc.run_episode()
        weight_update_size = np.linalg.norm(np.subtract(mc.weights, prev_weight_vector))
        print(f"Running episode {i+1}, magnitude of weight update is {weight_update_size}")
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    return mc.weights

# Normalize both approximation and true value function, and plot them and generate statistics (TBD)
def compare_value_funcs():
    pass

# Plot both approx and true value function
def plot_value_funcs(q_solver, approx_weights, true_weights):
    fig, axs = plt.subplots(1, 2)
    
    params = {"mass":1, "length":1, "gravity":1}
    initial_state = np.array([0.1, 0])
    def policy(s):
        return 0
    env = InvertedPendulum(params=params, initial_state=initial_state, sim_timestep = 0.002, control_timestep= 0.1)
    mc = MonteCarloContinuous(initial_state, env, discount_factor=0, policy=policy, max_timestep=100)
    mc.weights = true_weights
    
    x_min = -1 * np.pi                 # Limits of sampled state-space (x=pos, y=vel)
    x_max = np.pi
    y_min = -1
    y_max = 1
    res = 100                          # Resolution of state-space to sample
    X = np.linspace(x_min, x_max, res) # State-space coordinates
    Y = np.linspace(y_min, y_max, res)
    Z = np.zeros((res, res))
    a = 0                              # Array coordinates
    b = 0
    for x in X:
        for y in Y:
            Z[b, a] = mc.state_value_approx((x, y))
            b += 1
        b = 0
        a += 1

    img = axs[0].imshow(Z, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    fig.colorbar(img, ax=axs[0])
    axs[0].set_aspect((x_max - x_min) / (y_max - y_min))
    axs[0].set_xlabel("pos")
    axs[0].set_ylabel("vel")
    axs[0].set_title("True value function")
    
    q_solver.weights = approx_weights

    max_torque = env.params["mass"] * env.params["gravity"] * env.params["length"]
    X = np.linspace(x_min, x_max, res) # State-action-space coordinates
    Y = np.linspace(y_min, y_max, res)
    max_q_vals = np.zeros((res, res))
    max_q_actions = np.zeros((res, res))
    T = np.linspace(-1 * max_torque, max_torque, res)
    a = 0   # Array coordinates
    b = 0
    for x in X:
        for y in Y:
            max_q_val = q_solver.action_value_approx(T[0], (a, b))
            max_q_action = T[0]
            for t in T:
                q_val = q_solver.action_value_approx(t, (a, b))
                if (q_val > max_q_val):
                    max_q_val = q_val
                    max_q_action = t
            max_q_vals[b, a] = max_q_val
            max_q_actions[b, a] = max_q_action
            b += 1
        b = 0
        a += 1
    
    img = axs[1].imshow(max_q_vals, origin="lower", extent=[x_min, x_max, y_min, y_max])
    axs[1].set_title("Approximated value function\n(Maximum Q-value for each state)")
    fig.colorbar(img, ax=axs[1])
    axs[1].set_aspect((x_max - x_min) / (y_max - y_min))
    axs[1].set_xlabel("pos")
    axs[1].set_ylabel("vel")
    
    plt.show()

q_solver = initialize_env_and_solver(sim_timestep=0.01, control_timestep=0.1)
approx_weights = get_approx_weights(q_solver, num_episodes=200)
true_weights = get_true_weights(q_solver)
plot_value_funcs(q_solver, approx_weights, true_weights)