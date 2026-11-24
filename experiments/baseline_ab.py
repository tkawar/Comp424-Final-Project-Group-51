"""A stand-in for the course's "simple alpha-beta baseline with heuristic evaluation".

Not part of the submission. It is a plain iterative-deepening alpha-beta on the course helpers,
with no move ordering and no transposition table, and a line-count evaluation. Use it as an
opponent:  python simulator.py --player_1 student_agent --player_2 experiments/baseline_ab.py
"""
from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, execute_move, check_winner, copy_state, top_view, LINES
import constants
from constants import STEP_TIME_LIMIT, OBSTACLE
import time

_BUDGET = 0.8                     # share of STEP_TIME_LIMIT used
_MAX_DEPTH = 64                   # set to 2 or 3 for a fixed-depth baseline
_WIN = 100000


class _Timeout(Exception):
    pass


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "BaselineAB"
        self.lines = LINES[constants.WIN_LENGTH]
        self.last_depth = 0

    def evaluate(self, board, player):
        owners = top_view(board)[0].tolist()
        score = 0
        for line in self.lines:
            mine = theirs = 0
            dead = False
            for r, c in line:
                o = owners[r][c]
                if o == player:
                    mine += 1
                elif o == OBSTACLE:
                    dead = True
                    break
                elif o:
                    theirs += 1
            if dead:
                continue
            if theirs == 0:
                score += mine * mine
            elif mine == 0:
                score -= theirs * theirs
        return score

    def search(self, board, reserves, player, depth, alpha, beta, ply):
        if time.perf_counter() > self.deadline:
            raise _Timeout
        best = -10 * _WIN
        best_move = None
        for mv in get_valid_moves(board, reserves, player):
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv)
            w = check_winner(b, player)
            if w == player:
                v = _WIN - ply
            elif w:
                v = -(_WIN - ply)
            elif depth <= 1:
                v = self.evaluate(b, player)
            else:
                v = -self.search(b, r, 3 - player, depth - 1, -beta, -alpha, ply + 1)[0]
            if v > best:
                best, best_move = v, mv
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        if best_move is None:
            return -(_WIN - ply), None
        return best, best_move

    def step(self, board, reserves, player, opponent):
        self.deadline = time.perf_counter() + STEP_TIME_LIMIT * _BUDGET
        move = get_valid_moves(board, reserves, player)[0]
        self.last_depth = 0
        try:
            for depth in range(1, _MAX_DEPTH + 1):
                v, mv = self.search(board, reserves, player, depth, -10 * _WIN, 10 * _WIN, 0)
                if mv is not None:
                    move = mv
                self.last_depth = depth
                if abs(v) > _WIN - 1000:
                    break
        except _Timeout:
            pass
        return move
