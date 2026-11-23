"""Student agent: edit ONLY this file."""

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES)
#added a new import "LINES"
from constants import *
import numpy as np
import time

# copied from greedy 
def after(board, reserves, player, move):
    """Return the position after `player` plays `move`. The original arrays are not changed.

    Takes:   board, reserves - the position before the move.
             player          - the player who moves, 1 or 2.
             move            - a legal Move for that player.
    Returns: (board, reserves), new arrays with the move played.
    Example: b2, r2 = after(board, reserves, 1, Move(None, (0, 0), LARGE))
    """
    b, r = copy_state(board, reserves)
    execute_move(b, r, player, move)
    return b, r

def eval(board, reserves, player):
    """Evaluate the position, return a number: higher is better for `player`."""
    opponent = 3-player #since its player 1 or 2, subtracting from 3 gives the other player
    owners, _ = top_view(board) #as a start we ignore the size, 
    #top_view returns two arrays, one for owners and one for sizes

    score = 0
    weights = (0,1,10, 100, 1000) #weights need to be fine tuned later

    for line in LINES[WIN_LENGTH]:
        #used LINES as it loops through all lines of length 4 (WIN_LENGTH) in the grid
        cells = [owners[i,j] for i,j in line]
        if OBSTACLE in cells:
            #we cant win on a line with an obstacle
            continue

        my_count = sum(owner == player for owner in cells)
        opp_count = sum(owner == opponent for owner in cells)

        if my_count > 0 and opp_count == 0:
            score += weights[my_count]
        elif opp_count > 0 and my_count == 0:
            score -= weights[opp_count] 
    
    return score

def order_moves(board, reserves, player, moves, perferred_move):
    """Return a list of moves in order of preference. This is a placeholder for now."""
    #orders moves by promising moves first

    return get_valid_moves(board, reserves, player)

def minimax(board, reserves, player_to_move, root_player, depth, ply, alpha, beta, deadline):
    """Return the best move for `root_player` using minimax with alpha-beta pruning."""

    #this is VSCode generated, ill fix it up later, 
    # place holder for now

    #check for time limit
    #check for win, loss, draw
    # at depth zero, call eval(),
    # simulate moves and search recursively
    #update alpha and beta, prune if possible (alpha>= beta)
    #return score
    if time.perf_counter() >= deadline:
        raise TimeoutError("Time limit exceeded in minimax")

    game_over, winner, _ = check_endgame(board, reserves, player_to_move, ply)
    if game_over:
        if winner == root_player:
            return 10000000
        elif winner == 0: 
            return 0
        else:
            return -10000000

    if depth == 0:
        return eval(board, reserves, root_player)

    moves = get_valid_moves(board, reserves, player_to_move)

    turn = player_to_move == root_player

    if turn:
        best_score = float("-inf")
    else: 
        best_score = float("inf")


    for move in moves:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Time limit exceeded in minimax")
        
        next_board, next_reserves = after(board, reserves, player_to_move, move) 
        score = minimax(next_board, next_reserves, 3 - player_to_move, root_player, depth - 1, ply+1, beta, alpha, deadline)
        if turn:
            best_score = max(best_score, score)
        else:
            best_score = min(best_score, score)

    return best_score

    # will implememnt this later, for now we will use the eval function to score moves
    #     if score > best_score:
    #         best_score = score
    #         best_move = move

    #     alpha = max(alpha, score)
    #     if beta <= alpha:
    #         break

    # return best_move, best_score

def search(board, reserves, player, depth, ply, deadline, preferred_move=None):
    """Return the best move for `player` using minimax with alpha-beta pruning."""
    # VSCdode generated, ill fix it up later, place holder for now
    return 

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.rng = np.random.default_rng() # randomly selected to keep a fall back step
        self.turn_count = 0 #keep track of the turn count to adjust depth of search

    # def step(self, board, reserves, player, opponent):
    #     """Return the Move you want to play.

    #     Takes: board, reserves - your own copies of the current position.
    #            player, opponent - your player number (1 or 2) and the other one.
    #     Agent.step in agents/agent.py explains each argument and the time limit.
    #     """
    #     start = time.perf_counter()
    #     # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
    #     # helpers.get_valid_moves / check_winner already respect them; your own
    #     # move generator and evaluation function must too.
    #     # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
    #     return random_move(board, reserves, player)

    
    # copied from greedy and modified to include time limit and fallback move
    def step(self, board, reserves, player, opponent):
        #after implementing each of the above, fix up step() to use them.
        deadline = time.perf_counter() + STEP_TIME_LIMIT - 0.2
        depth_limit = 2

        ply = 2 * self.turn_count + player -1
        self.turn_count += 1

        moves = get_valid_moves(board, reserves, player)
        best_move = moves[int(self.rng.integers(len(moves)))] #prev named fallback
        best_score = float("-inf")


        try:
            for move in moves:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Time limit exceeded in step")

                next_board, next_reserves = after(
                    board, reserves, player, move
                )

                score = minimax(next_board, next_reserves, opponent, player, depth_limit-1, ply+1, float("-inf"), float("inf"), deadline)
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score == 10000000:
                    break

        except TimeoutError:
            pass
        
        return best_move
    