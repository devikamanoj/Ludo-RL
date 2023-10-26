import numpy as np
from qTable import Rewards
from stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace): #inherits from the state space class
    ai_player_idx = -1 #index of the AI player
    debug = False #controls whether or not the agent prints debug statements
    q_learning = None #stores the q-learning algorithm used by the agent
    state = None #current stae of  the agent
    action = None #stores the actions that the agent took in the previous state

    def __init__(self, ai_player_idx, gamma=0.3, learning_rate=0.2):
        super().__init__()
        # initialize the q-learning algorithm with the number of states and actions, discount factor and the learning rate
        self.q_learning = Rewards(len(State), len(Action), gamma=gamma, lr=learning_rate)
        # set the index of the AI player in the list of the players
        self.ai_player_idx = ai_player_idx

    def update(self, players, pieces_to_move, dice): #updates the agent's statet and action
        #the agent first  updates its state based on the current state of the game. It then chooses the next action to take using the Q-learning algorithm.
        super().update(players, self.ai_player_idx, pieces_to_move, dice)
        action_table = self.action_table_player.get_action_table() #gets the action table for the AI player
        state, action = self.q_learning.choose_next_action(self.ai_player_idx, action_table)#chooses the next action to take using the Q-learning algorithm
        pieces_to_move = self.action_table_player.get_piece_to_move(state, action) #gets the pieces to move for the given state and action
        self.state = state #sets the current state of the agent
        self.action = action #sets the action that the agent took in the previous state
        return pieces_to_move #returns the pieces to move for the AI player

    def reward(self, players, pieces_to_move): #reward the agent for taking an action
        super().get_possible_actions(players, self.ai_player_idx, pieces_to_move)# calls the method from the statespace class to get the list of possible actions for the AI player
        new_action_table = np.nan_to_num(self.action_table_player.get_action_table(), nan=0.0) #convert action table to numpy
        self.q_learning.reward(self.state, new_action_table, self.action)
