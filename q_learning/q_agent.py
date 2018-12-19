import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.losses = list()

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

        # append normalized losses to the list
        self.losses.append(abs(new_q - current_q) / 100)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


# our custom list to store losses history
history = {'losses': []}


def plot_loss(history):
    f, ax = plt.subplots(figsize=(7, 4))
    f.canvas.set_window_title('Result')

    ax.set_title('Training Loss')
    ax.set_xlabel('episodes')

    loss = history['losses']
    episodes = range(len(loss))

    ax.plot(episodes, loss, 'r', label='loss')

    ax.legend()
    plt.tight_layout()
    plt.show()


def check_if_have_none_or_more_then_two_argument():
    return len(sys.argv) < 2 or len(sys.argv) > 2


def check_if_argument_value_invalid():
    return sys.argv[1] != 'i' and sys.argv[1] != 'ii' and sys.argv[1] != 'iii' and sys.argv[1] != 'iv'


def exit_and_print_error():
    sys.exit('You should specify one argument: i, ii, iii, or iv')


if __name__ == "__main__":
    if check_if_have_none_or_more_then_two_argument() or check_if_argument_value_invalid():
        exit_and_print_error()
    else:
        scenario = sys.argv[1]
        env = Env()
        agent = QLearningAgent(actions=list(range(env.n_actions)))
        run_time = time.time()

        for episode in range(500):
            start_time = time.time()
            state = env.reset()
            agent.losses.clear()

            # For decreasing epsilon case
            if scenario == "ii":
                if episode == 100:
                    agent.epsilon = 0.075
                elif episode == 200:
                    agent.epsilon = 0.05
                elif episode == 300:
                    agent.epsilon = 0.025
                elif episode == 400:
                    agent.epsilon = 0

            while True:
                env.render()

                # take action and proceed one step in the environment
                action = agent.get_action(str(state))
                next_state, reward, done = env.step(action)

                # with sample <s,a,r,s'>, agent learns new q function
                agent.learn(str(state), action, reward, str(next_state))

                state = next_state
                env.print_value_all(agent.q_table)

                # if episode ends, then break
                if done:
                    elapsed_time = time.time() - start_time
                    print('episode : %s --- time : %s' % (episode, round(elapsed_time, 0)))
                    # save the loss to history
                    history["losses"].append(np.mean(agent.losses))
                    break

        run_elapsed_time = time.time() - run_time
        print('Running Time : %s' % (round(run_elapsed_time, 0)))
        plot_loss(history)
