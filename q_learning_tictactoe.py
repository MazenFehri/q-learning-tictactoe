import numpy as np
import random

board = np.array([['-', '-', '-'],
                  ['-', '-', '-'],
                  ['-', '-', '-']])

players = ['X', 'O']
num_players = len(players)
Q = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.5
num_episodes = 10000

def print_board(board):
    for row in board:
        print('  |  '.join(row))
        print('---------------')

def board_to_string(board):
    return ''.join(board.flatten())

def is_game_over(board):
    for row in board:
        if row[0] == row[1] == row[2] != '-':
            return True, row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != '-':
            return True, board[0][col]

    if board[0][0] == board[1][1] == board[2][2] != '-':
        return True, board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '-':
        return True, board[0][2]

    if all(cell != '-' for row in board for cell in row):
        return True, 'draw'

    return False, None

def choose_action(board, exploration_rate):
    state = board_to_string(board)

    if random.uniform(0, 1) < exploration_rate or state not in Q:
        empty_cells = np.argwhere(board == '-')
        action = tuple(random.choice(empty_cells))
    else:
        q_values = Q[state]
        empty_cells = np.argwhere(board == '-')
        empty_q_values = [q_values[cell[0], cell[1]] for cell in empty_cells]
        max_q_value = max(empty_q_values)
        max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]
        max_q_index = random.choice(max_q_indices)
        action = tuple(empty_cells[max_q_index])

    return action

def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state, np.zeros((3, 3)))

    next_q_values = Q.get(board_to_string(next_state), np.zeros((3, 3)))
    max_next_q_value = np.max(next_q_values)

    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])

    Q[state] = q_values

def train_agent():
    global exploration_rate
    agent_wins = 0
    num_draws = 0

    for episode in range(num_episodes):
        board = np.array([['-', '-', '-'],
                          ['-', '-', '-'],
                          ['-', '-', '-']])

        current_player = random.choice(players)
        game_over = False

        while not game_over:
            action = choose_action(board, exploration_rate)

            row, col = action
            board[row, col] = current_player

            game_over, winner = is_game_over(board)

            if game_over:
                if winner == current_player:
                    reward = 1
                    agent_wins += 1
                elif winner == 'draw':
                    reward = 0
                    num_draws += 1
                else:
                    reward = -1
                update_q_table(board_to_string(board), action, board, reward)
            else:
                current_player = players[(players.index(current_player) + 1) % num_players]

            if not game_over:
                next_state = board.copy()
                next_state[action[0], action[1]] = players[(players.index(current_player) + 1) % num_players]
                update_q_table(board_to_string(board), action, next_state, 0)

        exploration_rate *= 0.99

    return agent_wins, num_draws

def play_game():
    board = np.array([['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']])

    current_player = random.choice(players)
    game_over = False

    while not game_over:
        print_board(board)
        if current_player == 'X':
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))
            action = (row, col)
        else:
            action = choose_action(board, exploration_rate=0)

        row, col = action
        board[row, col] = current_player

        game_over, winner = is_game_over(board)

        if game_over:
            print_board(board)
            if winner == 'X':
                print("Human player wins!")
            elif winner == 'O':
                print("Agent wins!")
            else:
                print("It's a draw!")
        else:
            current_player = players[(players.index(current_player) + 1) % num_players]

agent_wins, num_draws = train_agent()

play_game()

print(f"Agent win percentage: {agent_wins / num_episodes * 100:.2f}%")
print(f"Draw percentage: {num_draws / num_episodes * 100:.2f}%")
