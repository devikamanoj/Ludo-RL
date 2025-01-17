import ludopy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from player import QLearningAgent


# win rate of AI agent over a certain window of episodes is calculated using moving averages
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    # same convolution (product of moving windows) of interval and window
    return np.convolve(interval, window, 'same')

def epsilon_decay(epsilon, decay_rate, episode):
    # gradually reduce the level of exploration over time as the agent gains more experience
    # higher e = more exploration
    return epsilon * np.exp(-decay_rate*episode)

def start_teaching_ai_agent(episodes, no_of_players, epsilon, epsilon_decay_rate, lr, gamma):

    # Houskeeping variables
    ai_player_winning_avg = [] # 0 for AI losing, 1 for AI winning
    epsilon_list = [] # val of e over episodes
    idx = [] # ep indices
    ai_player_won = 0

    # Store data
    win_rate_list = []
    max_expected_return_list = []

    if no_of_players == 4:
        g = ludopy.Game(ghost_players=[])
    elif no_of_players == 3:
        g = ludopy.Game(ghost_players=[1])
    else:
        g = ludopy.Game(ghost_players=[1,3])

    ai_player_1 = QLearningAgent(0, learning_rate=lr, gamma=gamma)

    for episode in range(0, episodes):

        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = g.get_observation()
            
            # move_pieces: pieces the player can move
            if len(move_pieces):
                # AI player takes an action. If it is not AI, then move randomly (showing human interaction)
                if ai_player_1.ai_player_idx == player_i:
                    piece_to_move = ai_player_1.update(g.players, move_pieces, dice)
                    if not piece_to_move in move_pieces:
                        # to not allow any moves
                        g.render_environment()
                else:
                    # emulate human movement by making moves random
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            
            if episode > 1:
                board = g.render_environment()
                cv2.imshow("Ludo Board", board)
                cv2.waitKey(10)
                
            if ai_player_1.ai_player_idx == player_i and piece_to_move != -1:
                # if AI player moved, update reward
                ai_player_1.reward(g.players, [piece_to_move])
        if episode == 500:
            g.save_hist_video("game.mp4")
            
        new_epsilon_after_decay = epsilon_decay(epsilon=epsilon, decay_rate=epsilon_decay_rate,episode=episode)
        epsilon_list.append(new_epsilon_after_decay)
        ai_player_1.q_learning.update_epsilon(new_epsilon_after_decay)


        if g.first_winner_was == ai_player_1.ai_player_idx:
            ai_player_winning_avg.append(1)
            ai_player_won = ai_player_won + 1
        else:
            ai_player_winning_avg.append(0)

        idx.append(episode)

        # Print some results
        win_rate = ai_player_won / len(ai_player_winning_avg)
        win_rate_percentage = win_rate * 100
        win_rate_list.append(win_rate_percentage)

        if episode % 100 >= 0:
            print("Episode: ", episode)
            print(f"Win rate: {np.round(win_rate_percentage,1)}%")
    
        # append and reset max expected reward
        max_expected_return_list.append(ai_player_1.q_learning.max_expected_reward)
        ai_player_1.q_learning.max_expected_reward = 0


    # Moving averages
    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rate_list, 0, 0)) # sum of all win rates up to a specific position in the list
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size # cumulative sum of data points in moving average

    cumsum_vec = np.cumsum(np.insert(max_expected_return_list, 0, 0)) # cumulative sum of max expectations
    max_expected_return_list_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    
    moving_average_list = [0] * window_size 
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list + max_expected_return_list_ma.tolist()
    return win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma


# FINAL AGENT PLAYING AGAINST 1,2 and 3 RANDOM PLAYERS
learning_rate = 0.2
gamma = 0.5
epsilon = 0.9
epsilon_decay_rate = 0.05
episodes = 500
no_of_players = np.random.randint(1,4)

# Start teaching the agent
win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma = start_teaching_ai_agent(episodes, no_of_players, epsilon, epsilon_decay_rate, learning_rate, gamma)

# Plot win rates against opponents
fig, axs = plt.subplots(1)
axs.set_title("Win Rate against different number of opponents")
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.plot(win_rate_list)
if(no_of_players==4):
    axs.set_title(['Win Rate against 3 opponents'])
elif(no_of_players==3):
    axs.set_title(['Win Rate against 2 opponents'])
else:
    axs.set_title(['Win Rate against 1 opponent'])
    
# Plot epsilon decay
fig, axs = plt.subplots(1)
axs.set_title("Epilson Decay")
axs.set_xlabel('Episodes')
axs.set_ylabel('Epsilon')
axs.plot(epsilon_list, color='tab:red')
axs.legend(['Epsilon Decay 0.05'])

plt.show()
