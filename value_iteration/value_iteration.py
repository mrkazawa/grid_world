# -*- coding: utf-8 -*-
from environment import GraphicDisplay, Env
import sys
import random


class ValueIteration:
    def __init__(self, env, scenario):
        self.env = env
        # 2-d list for the value function
        if scenario == 'ii':
            self.value_table = self.generate_random_value_table(
                env.width, env.height)
        else:
            self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def generate_random_value_table(self, width, height):
        value_table = [[round(random.random(), 2)] *
                       width for _ in range(height)]
        return value_table

    # get next value function table from the current value function table
    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width
                            for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            value_list = []

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))

            # return the maximum value(it is the optimality equation!!)
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)

        self.value_table = next_value_table

    # get action according to the current value function table
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # calculating q values for the all actions and
        # append the action to action list which has maximum q value
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

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
        value_iteration = ValueIteration(env, scenario)
        grid_world = GraphicDisplay(value_iteration, scenario)
        grid_world.mainloop()
