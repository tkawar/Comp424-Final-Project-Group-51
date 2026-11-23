"""Student agent: edit ONLY this file."""
from shutil import move

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
        #used LINES as it loops through all lines of length 4 in the grid
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

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.rng = np.random.default_rng() # randomly selected to keep a fall back step

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
        deadline = time.perf_counter() + STEP_TIME_LIMIT - 0.2

        moves = get_valid_moves(board, reserves, player)
        fallback = moves[int(self.rng.integers(len(moves)))]

        #check every move for an immediate win.
        candidates = []

        for move in moves:
            if time.perf_counter() >= deadline:
                return fallback

            next_board, next_reserves = after(
                board, reserves, player, move
            )

            score = eval(next_board, next_reserves, player)

            winner = check_winner(next_board, player)

            if winner == player:
                return move

            # avoid moves that immediately give opp a win.
            if winner != opponent:
                candidates.append((move, next_board, next_reserves))

        #look for moves that allow no immediate winning reply.
        safe = []
        best_move = None
        best_score = float("-inf") 
        for move, next_board, next_reserves in candidates:
            if time.perf_counter() >= deadline:
                break

            opponent_can_win = False

            replies = get_valid_moves(
                next_board, next_reserves, opponent
            )

            for reply in replies:
                if time.perf_counter() >= deadline:
                    # This candidate has not been fully checked.
                    return best_move if best_move is not None else fallback

                reply_board, _ = after(
                    next_board, next_reserves, opponent, reply
                )

                if check_winner(reply_board, opponent) == opponent:
                    opponent_can_win = True
                    break

            if not opponent_can_win:
                safe.append(move)

                score = eval(next_board, next_reserves, player)

                if score > best_score:
                    best_score = score
                    best_move = move

        if best_move is not None:
            return best_move
           
        

        # favor a move that does not immediately expose a losing line.
        if candidates:
            return candidates[int(self.rng.integers(len(candidates)))][0]

        return fallback
    