"""Student agent: edit ONLY this file.

Iterative-deepening alpha-beta (negamax) search with a line-based evaluation.
"""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, state_key)
from constants import *
import numpy as np
import time

WIN_SCORE = 1_000_000
TIME_MARGIN = 0.5            # seconds kept free below STEP_TIME_LIMIT
_LINES_CACHE = {}


class SearchTimeout(Exception):
    """Raised inside the search when the deadline has passed."""


def after(board, reserves, player, move):
    """Return the position after `player` plays `move`. The original arrays are not changed."""
    b, r = copy_state(board, reserves)
    execute_move(b, r, player, move)
    return b, r


def get_lines(rows, cols, k):
    """All windows of length k (rows, columns, both diagonals) as index tuples, cached."""
    key = (rows, cols, k)
    if key in _LINES_CACHE:
        return _LINES_CACHE[key]
    lines = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = r + dr * (k - 1), c + dc * (k - 1)
                if 0 <= end_r < rows and 0 <= end_c < cols:
                    rr = tuple(r + dr * i for i in range(k))
                    cc = tuple(c + dc * i for i in range(k))
                    lines.append((rr, cc))
    _LINES_CACHE[key] = lines
    return lines


def evaluate(board, player, opponent):
    """Heuristic score from `player`'s point of view (higher is better).
    top_view returns (owners, sizes); owners is 4x4 with 0 empty, 1/2 a player, 3 obstacle."""
    top, _ = top_view(board)
    rows, cols = top.shape
    score = 0
    for rr, cc in get_lines(rows, cols, WIN_LENGTH):
        cells = top[rr, cc]
        mine = int(np.count_nonzero(cells == player))
        theirs = int(np.count_nonzero(cells == opponent))
        if mine and not theirs:
            score += 10 ** mine
        elif theirs and not mine:
            score -= 10 ** theirs
    center = top[rows // 2, cols // 2]
    if center == player:
        score += 3
    elif center == opponent:
        score -= 3
    return score


@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.rng = np.random.default_rng()
        self.best_reply = {}      # state_key -> best move found there (for move ordering)
        self.deadline = 0.0
        self.last_depth = 0       # deepest completed depth on the last step (for debugging)

    # ------------------------------------------------------------ search
    def ordered_moves(self, board, reserves, to_move):
        """Legal moves, with the best move remembered from an earlier search first."""
        moves = get_valid_moves(board, reserves, to_move)
        best = self.best_reply.get(state_key(board, reserves, to_move))
        if best is not None and best in moves:
            moves.remove(best)
            moves.insert(0, best)
        return moves

    def negamax(self, board, reserves, to_move, other, depth, alpha, beta, ply):
        """Score of the position for `to_move`, searched `depth` plies deep."""
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout
        if depth == 0:
            return evaluate(board, to_move, other)

        moves = self.ordered_moves(board, reserves, to_move)
        if not moves:
            return 0

        best_score = -float("inf")
        best_move = moves[0]
        for move in moves:
            child_b, child_r = after(board, reserves, to_move, move)
            winner = check_winner(child_b, to_move)
            if winner == to_move:
                score = WIN_SCORE - ply           # faster wins score higher
            elif winner == other:
                score = -WIN_SCORE + ply          # uncovered the opponent's line
            else:
                score = -self.negamax(child_b, child_r, other, to_move,
                                      depth - 1, -beta, -alpha, ply + 1)
            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break                             # beta cut-off
        self.best_reply[state_key(board, reserves, to_move)] = best_move
        return best_score

    # ------------------------------------------------------------ step
    def step(self, board, reserves, player, opponent):
        self.deadline = time.perf_counter() + STEP_TIME_LIMIT - TIME_MARGIN

        moves = get_valid_moves(board, reserves, player)
        best_move = moves[int(self.rng.integers(len(moves)))]   # fallback

        # Immediate win: take it without searching.
        for move in moves:
            b, _ = after(board, reserves, player, move)
            if check_winner(b, player) == player:
                return move

        depth = 1
        self.last_depth = 0
        while depth <= MAX_PLIES:
            try:
                score = self.negamax(board, reserves, player, opponent,
                                     depth, -float("inf"), float("inf"), 0)
            except SearchTimeout:
                break
            # A full depth finished: its best move is now trustworthy.
            best_move = self.best_reply.get(state_key(board, reserves, player), best_move)
            self.last_depth = depth
            if abs(score) >= WIN_SCORE - MAX_PLIES:
                break                              # forced win or loss found; deeper won't change it
            depth += 1

        return best_move

