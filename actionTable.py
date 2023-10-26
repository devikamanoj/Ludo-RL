import numpy as np


class ActionTableEntry(): #used to store the action table entries for a given state and action pair 
    def __init__(self, piece, value):
        super().__init__()
        self.pice = piece #piece to move
        self.value = value #value of the action

    def add_entry(self, piece, value): #add an entry to the action table
        self.__piece.append(piece)
        self.__value.append(value)


class ActionTable():
    action_table = None
    state = 0

    def __init__(self, states, actions):# defines constructor method for action table
        super().__init__()
        self.states = states #number of states
        self.actions = actions #number of actions
        self.reset()

    def set_state(self, state): #set the current state of the action table
        self.state = state.value

    def get_action_table(self): #returns the action table(numpy array that stores the action table)
        return self.action_table

    def get_piece_to_move(self, state, action):
        #returns the piece to move for the given state and action
        #returns -1 if the state or action is invalid
        if state < 0 or action < 0:
            return -1
        return int(self.piece_to_move[state, action])

    def reset(self): #returns table to the initial state (all values are set to nan)
        self.action_table = np.full((self.states, self.actions), np.nan)
        self.piece_to_move = np.full((self.states, self.actions), np.nan)

    # update the action table with the given action and piece to move
    def update_action_table(self, action, piece, value):
        if np.isnan(self.action_table[self.state, action.value]): 
            #checks if the action table entry for the given state and action is NaN. If it is, then the action table entry has not yet been initialized and needs to be initialized.
            self.action_table[self.state, action.value] = 1
            self.piece_to_move[self.state, action.value] = piece