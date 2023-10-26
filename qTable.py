import os.path
import random

import numpy as np
from stateSpace import Action
class Rewards():

    rewards_table = np.zeros(len(Action)) #created an array filled with zeros with length equal to the number of actions
    q_table = None #used for storing q-values in the q-learning algorithm
    epoch = 0
    iteration = 0

    def __init__(self, states, actions, epsilon=0.9, gamma=0.3, lr=0.2, learning=True): #defines constructor method for rewards
        super().__init__() #to call the supreclass constructor
        self.learning = learning # used to determine whether the agent is learning or not
        self.q_table = np.zeros([states, actions]) #creates a q-table with the number of states and actions as zeros
        self.epsilon_greedy = epsilon #epsilon is used to determine the probability of exploration vs exploitation in the decision making process
        self.gamma = gamma #gamma is the discount factor used in the q-learning algorithm
        self.lr = lr #learning rate used in the q-learning algorithm

        self.max_expected_reward = 0 #used to store the maximum expected reward for a given state
        
        # Rewards table
        VERY_BAD = -0.8
        BAD = -0.4
        GOOD = 0.4
        VERY_GOOD = 1.2

        # Rewards table for each move
        self.rewards_table[Action.SAFE_MoveOut.value] = 0.4
        self.rewards_table[Action.SAFE_MoveDice.value] = 0.01
        self.rewards_table[Action.SAFE_Goal.value] = 0.8
        self.rewards_table[Action.SAFE_Star.value] = 0.8
        self.rewards_table[Action.SAFE_Globe.value] = 0.4
        self.rewards_table[Action.SAFE_Protect.value] = 0.2
        self.rewards_table[Action.SAFE_Kill.value] = 1.5
        self.rewards_table[Action.SAFE_Die.value] = -0.5
        self.rewards_table[Action.SAFE_GoalZone.value] = 0.2

        self.rewards_table[Action.UNSAFE_MoveOut.value] = self.rewards_table[Action.SAFE_MoveOut.value] + BAD
        self.rewards_table[Action.UNSAFE_MoveDice.value] = self.rewards_table[Action.SAFE_MoveDice.value] + BAD
        self.rewards_table[Action.UNSAFE_Star.value] = self.rewards_table[Action.SAFE_Star.value] + BAD
        self.rewards_table[Action.UNSAFE_Globe.value] = self.rewards_table[Action.SAFE_Globe.value] + GOOD
        self.rewards_table[Action.UNSAFE_Protect.value] = self.rewards_table[Action.SAFE_Protect.value] + GOOD
        self.rewards_table[Action.UNSAFE_Kill.value] = self.rewards_table[Action.SAFE_Kill.value] + GOOD
        self.rewards_table[Action.UNSAFE_Die.value] = self.rewards_table[Action.SAFE_Die.value] + VERY_BAD
        self.rewards_table[Action.UNSAFE_GoalZone.value] = self.rewards_table[Action.SAFE_GoalZone.value] + GOOD
        self.rewards_table[Action.UNSAFE_Goal.value] = self.rewards_table[Action.SAFE_Goal.value] + GOOD

        self.rewards_table[Action.HOME_MoveOut.value] = self.rewards_table[Action.SAFE_MoveOut.value] + VERY_GOOD
        self.rewards_table[Action.HOME_MoveDice.value] = self.rewards_table[Action.SAFE_MoveDice.value] + VERY_BAD
        self.rewards_table[Action.HOME_Star.value] = self.rewards_table[Action.SAFE_Star.value] + VERY_BAD
        self.rewards_table[Action.HOME_Globe.value] = self.rewards_table[Action.SAFE_Globe.value] + VERY_BAD
        self.rewards_table[Action.HOME_Protect.value] = self.rewards_table[Action.SAFE_Protect.value] + VERY_BAD
        self.rewards_table[Action.HOME_Kill.value] = self.rewards_table[Action.SAFE_Kill.value] + VERY_BAD
        self.rewards_table[Action.HOME_Die.value] = self.rewards_table[Action.SAFE_Die.value] + VERY_BAD
        self.rewards_table[Action.HOME_GoalZone.value] = self.rewards_table[Action.SAFE_GoalZone.value] + VERY_BAD
        self.rewards_table[Action.HOME_Goal.value] = self.rewards_table[Action.SAFE_Goal.value] + VERY_BAD

    def update_epsilon(self, new_epsilon): #update the epsilon greedy variable
        self.epsilon_greedy = new_epsilon

    def get_state_action_of_array(self, value, array): #  takes a value and an array and returns a random state-action pair where the array matches the value
        if np.isnan(value):
            return (-1, -1)
        idx = np.where(array == value)
        random_idx = random.randint(0, len(idx[0]) - 1)
        state = idx[0][random_idx]
        action = idx[1][random_idx]
        return (state, action)

    def choose_next_action(self, player, action_table):#choose the next action for the player based on the q-table and an action table
        # It implements an epsilon-greedy strategy, where the agent chooses a random action with probability epsilon_greedy or selects the action with the highest Q-value.
        q_table_options = np.multiply(self.q_table, action_table)
    
        if random.uniform(0, 1) < self.epsilon_greedy:
            self.iteration = self.iteration + 1
            nz = action_table[np.logical_not(np.isnan(action_table))]
            randomValue = nz[random.randint(0, len(nz) - 1)]
            state, action = self.get_state_action_of_array(randomValue, action_table)
        else:
            maxVal = np.nanmax(q_table_options)
            if not np.isnan(maxVal):
                state, action = self.get_state_action_of_array(maxVal, q_table_options)
            else:
                nz = action_table[np.logical_not(np.isnan(action_table))]
                random_value = nz[random.randint(0, len(nz) - 1)]
                state, action = self.get_state_action_of_array(random_value, action_table)
        return (state, action)


    def reward(self, state, new_action_table, action):
        #calculates the reward for a given state-action pair and updates the Q-table based on the Q-learning algorithm. It computes the new Q-value for the state-action pair and updates the max_expected_reward attribute.
        state = int(state)
        action = int(action)

        # Q-learning equation
        reward = self.rewards_table[action]
        # Q-learning
        estimate_of_optimal_future_value = np.max(self.q_table * new_action_table)
        old_q_value = self.q_table[state, action]
        delta_q = self.lr * (reward + self.gamma * estimate_of_optimal_future_value - old_q_value)
        self.max_expected_reward += reward
        
        # Update the Q table from the new action taken in the current state
        self.q_table[state, action] = old_q_value + delta_q
    