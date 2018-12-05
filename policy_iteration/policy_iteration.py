# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env
import sys
import numpy as np


class PolicyIteration:
    def __init__(self, env, scenario):
        self.env = env
        # 2-d list for the value function
        if scenario == 'ii':
            self.value_table = self.generate_random_value_table(
                env.width, env.height)
        else:
            self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # list of random policy (up, down, left, right)
        if scenario == 'ii':
            self.policy_table = self.generate_random_policy_table(
                env.width, env.height)
        else:
            self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                 for _ in range(env.height)]
        # setting terminal state
        self.policy_table[2][2] = []
        self.discount_factor = 0.9

    def generate_random_value_table(self, width, height):
        value_table = [[round(random.random(), 2)] *
                       width for _ in range(height)]
        return value_table

    def generate_random_policy_table(self, width, height):
        r = list(np.random.dirichlet(np.ones(4), size=1).ravel())
        policy_table = [[r] * width for _ in range(height)]
        return policy_table

    def policy_evaluation(self):
        next_value_table = [[0.00] * self.env.width
                            for _ in range(self.env.height)]

        # Bellman Expectation Equation for the every states
        for state in self.env.get_all_states():
            value = 0.0
            # keep the value function of terminal states as 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            result = [0.0, 0.0, 0.0, 0.0]  # initialize the policy

            # for every actions, calculate
            # [reward + (discount factor) * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # We normally can't pick multiple actions in greedy policy.
                # but here we allow multiple actions with same max values
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # probability of action
            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # get action according to the current policy
    def get_action(self, state):
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # get policy of specific state
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)


def check_if_have_none_or_more_then_two_argument():
    return len(sys.argv) < 2 or len(sys.argv) > 2


def check_if_argument_value_invalid():
    return sys.argv[1] != 'i' and sys.argv[1] != 'ii' and sys.argv[1] != 'iii'


def exit_and_print_error():
    sys.exit('You should specify one argument: i, ii, or iii')


if __name__ == "__main__":
    if check_if_have_none_or_more_then_two_argument() or check_if_argument_value_invalid():
        exit_and_print_error()
    else:
        scenario = sys.argv[1]
        env = Env(scenario)
        policy_iteration = PolicyIteration(env, scenario)
        grid_world = GraphicDisplay(policy_iteration, scenario)
        grid_world.mainloop()
