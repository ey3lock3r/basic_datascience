import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

env = gym.make("MountainCar-v0").env

env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

print("state vector dim =", state_dim)
print("n_actions =", n_actions)

mc_agent = MLPClassifier(
    hidden_layer_sizes=(30, 30),
    activation='tanh',
    solver='adam'
)

# initialize agent to the dimension of state space and number of actions
s = [env.reset()] * n_actions
print(f'init={s}')
[mc_agent.partial_fit(s, i, i) for i in range(n_actions)]

def generate_session_mc(agent, t_max=1000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0

    s = env.reset()
    
    for t in range(t_max):

        # use agent to predict a vector of action probabilities for state :s:
        print(f'state={s}')
        probs = agent.predict_proba(s).reshape(n_actions,)

        assert probs.shape == (n_actions,), "make sure probabilities are a vector (hint: np.reshape)"
        
        # use the probabilities you predicted to pick an action
        # sample proportionally to the probabilities, don't just take the most likely action
        a = np.random.choice(n_actions, p=probs)
        # ^-- hint: try np.random.choice

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward

def select_elites_mc(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order 
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """

    reward_threshold = np.percentile(rewards_batch, percentile)
    
    elite_states, elite_actions = [], []

    [elite_states.append(states_batch[i]) or elite_actions.append(actions_batch[i]) for i in range(len(rewards_batch)) if rewards_batch[i] > reward_threshold]
#     [elite_actions.extend(actions_batch[i]) for i in range(len(rewards_batch)) if rewards_batch[i] >= reward_threshold]
    
    return elite_states, elite_actions

dummy_states, dummy_actions, dummy_reward = generate_session_mc(mc_agent, t_max=5)
print("states:", np.stack(dummy_states))
print("actions:", dummy_actions)
print("reward:", dummy_reward)