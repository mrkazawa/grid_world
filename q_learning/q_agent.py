import numpy as np
import random
import time
import sys
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

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

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


def check_if_have_none_or_more_then_two_argument():
    return len(sys.argv) < 2 or len(sys.argv) > 2


def check_if_argument_value_invalid():
    return sys.argv[1] != 'i' and sys.argv[1] != 'ii' and sys.argv[1] != 'iii'


def exit_and_print_error():
    sys.exit('You should specify one argument: i, ii or iii')


if __name__ == "__main__":
    if check_if_have_none_or_more_then_two_argument() or check_if_argument_value_invalid():
        exit_and_print_error()
    else:
        scenario = sys.argv[1]
        env = Env(scenario)
        agent = QLearningAgent(actions=list(range(env.n_actions)))
        run_time = time.time()

        for episode in range(500):
            start_time = time.time()
            state = env.reset()

            # make epsilon bigger so that the agent can understand the env better
            if scenario == "iii":
                # if episode is high enough, make it full greedy
                # so we can easily capture the result
                if episode <= 100:
                    agent.epsilon = 0.5
                elif 100 < episode <= 200:
                    agent.epsilon = 0.4
                elif 200 < episode <= 300:
                    agent.epsilon = 0.2
                elif 300 < episode <= 400:
                    agent.epsilon = 0.1
                if episode > 400:
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
                    break

        run_elapsed_time = time.time() - run_time
        print('Running Time : %s' % (round(run_elapsed_time, 0)))
