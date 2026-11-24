"""Student agent: edit ONLY this file.

OVERVIEW
--------
This agent picks its move with a game-tree search: negamax with alpha-beta pruning, run with
iterative deepening. In plain words:

  1. Look at every move I can make, then every reply, then every reply to that, and so on.
  2. When the search has to stop looking deeper, give the position a score (the "evaluation").
  3. Assume both players always pick the move that is best for them, and choose the move that
     leads to the best score for me.

"Negamax" is minimax written with one function for both players: a position that is worth +5 to
me is worth -5 to my opponent, so each player just maximizes the negative of the other's score.

"Alpha-beta pruning" skips moves that cannot change the answer. If I already have a move worth 10
and I see that another move lets the opponent hold me to 3, I stop looking at that move.

"Iterative deepening" means we search 1 move deep, then 2, then 3, ... until the time runs out,
and play the best move of the last depth that finished. So there is always an answer ready.

HOW THE BOARD IS STORED (BITBOARDS)
-----------------------------------
The numpy board is too slow to copy at every step of the search, so the search converts it once
into six integers. Each integer is used as 16 on/off switches (bits), one per cell:

    cell (row, col)  ->  bit number row * 4 + col

    pieces[0], pieces[1], pieces[2]  = where player 1 has a small / medium / large piece
    pieces[3], pieces[4], pieces[5]  = the same for player 2

For example, if player 1 has large pieces on cells 0 and 5, pieces[2] is 0b100001.

BIT TRICKS USED BELOW
---------------------
    a | b             cells that are in a OR in b
    a & b             cells that are in a AND in b
    a & ~b            cells that are in a but NOT in b
    a ^ b             flips the cells of b inside a (used to put a piece down and pick it up)
    1 << n            the single cell number n
    x & -x            the lowest cell that is switched on in x
    b.bit_length()-1  the cell number of a single cell b
    x.bit_count()     how many cells are switched on in x

To visit every cell of a set x, the code repeats this loop:

    while x:
        bit = x & -x      # take the lowest cell
        x ^= bit          # remove it from the set
        ...use bit...

HOW A MOVE IS STORED (MOVE CODES)
---------------------------------
Inside the search a move is one small integer instead of a Move object:

    code = (size + 1)  +  (destination cell * 4)  +  ((source cell + 1) * 64)

The source part is 0 when the piece comes from the reserve. code_to_move() turns a code back
into a real Move at the end.
"""
from agents.agent import Agent
from store import register_agent
from helpers import Move, get_valid_moves, check_move_validity, LINES
import constants
from constants import BOARD_SIZE, PLAYER_1, PLAYER_2, OBSTACLE, MAX_PLIES, STEP_TIME_LIMIT
import gc
import time

NUM_CELLS = BOARD_SIZE * BOARD_SIZE     # 16 cells
ALL_CELLS = (1 << NUM_CELLS) - 1        # an integer with all 16 bits switched on

# ---------------------------------------------------------------- scores
WIN_SCORE = 1_000_000                   # a win right now. A win n moves away scores WIN_SCORE - n,
                                        # so the search prefers faster wins and slower losses.
WIN_ZONE = WIN_SCORE - 1000             # any score above this means "a forced win was found"
INFINITY = 10 * WIN_SCORE               # bigger than any real score

# ---------------------------------------------------------------- search settings
TIME_FRACTION = 0.75                    # use 75% of the 2 seconds (1.5 s), the rest is a safety margin
TABLE_MAX_SIZE = 300_000                # empty the transposition table when it has more entries than this
THREAT_EXTRA_DEPTH = 4                  # how many extra moves deep we may go while someone threatens to win
LATE_MOVE_START = 4                     # quiet moves after the first 4 are searched one move shallower
DRAW_PENALTY = 10                       # we count a draw as slightly bad for us, so we keep trying to win
NEAR_END_PLIES = 28                     # this close to the 100-move limit, the move number matters

# ---------------------------------------------------------------- evaluation weights
# These numbers were tuned by playing many games between versions of the agent.
TOP_PIECE_VALUE = 3                     # for each piece I have on top (visible) more than the opponent
MY_TURN_BONUS = 3                       # small bonus for being the player to move
LINE_VALUES = (0, 1, 4, 16, 0)          # value of a line where I have 0, 1, 2, 3 pieces on top
BLOCK_EIGHTHS = (8, 4, 3, 2, 2)         # ...multiplied by this / 8 when the opponent has 0, 1, 2, 3, 4
                                        # pieces in the same line (I would have to cover each of them)
BURIED_VALUE = 8                        # for each opponent piece trapped under one of my pieces
BURIED_MEDIUM_BONUS = 8                 # extra when the trapped piece is a medium under my large

# Only for testing: stop after this many positions instead of using the clock (0 = use the clock).
# With a fixed number of positions the agent plays the same moves on every run.
NODE_LIMIT = 0

# What kind of score is saved in the transposition table (see negamax).
EXACT = 0           # the true score of the position
AT_LEAST = 1        # the true score is this or higher (the search was cut short by pruning)
AT_MOST = 2         # the true score is this or lower (no move was better than alpha)

# The lookup tables are slow to build (about a second), so we keep them here after the first time.
SAVED_TABLES = {}   # win_length -> (has_line, completing_cells, line_masks)


class OutOfTime(Exception):
    """Raised inside the search when the time is up. step() then plays the best move found so far."""


def build_tables(win_length):
    """Build the lookup tables that make win checks fast. Returns (has_line, completing_cells, line_masks).

    A set of cells is an integer from 0 to 65535 (16 bits), so we can prepare the answer for every
    possible set once and then just look it up during the search.

    has_line[cells]         - 1 if `cells` contains a full winning line, otherwise 0.
    completing_cells[cells] - the cells that would finish a line if one more piece was added.
    line_masks              - every winning line as a set of cells.
    """
    if win_length in SAVED_TABLES:
        return SAVED_TABLES[win_length]

    # Turn every line from helpers.LINES (a list of (row, col) pairs) into a set of cells.
    line_masks = []
    for line in LINES[win_length]:
        mask = 0
        for row, col in line:
            mask |= 1 << (row * BOARD_SIZE + col)
        line_masks.append(mask)

    has_line = bytearray(1 << NUM_CELLS)
    completing_cells = [0] * (1 << NUM_CELLS)
    for mask in line_masks:
        other_cells = ALL_CELLS ^ mask          # the cells that are not part of this line

        # Any set of cells that contains the whole line is a win. We go through every possible
        # combination of the other cells. (sub - 1) & other_cells is a standard trick that steps
        # through all subsets of other_cells.
        sub = other_cells
        while True:
            has_line[mask | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & other_cells

        # Now the "almost" lines: the line with one cell missing. That missing cell completes it.
        remaining = mask
        while remaining:
            missing = remaining & -remaining
            remaining ^= missing
            line_without_one = mask ^ missing
            sub = other_cells
            while True:
                completing_cells[line_without_one | sub] |= missing
                if sub == 0:
                    break
                sub = (sub - 1) & other_cells

    SAVED_TABLES[win_length] = (has_line, completing_cells, line_masks)
    return SAVED_TABLES[win_length]


def build_line_scores(win_length):
    """Prepare the evaluation score of one line for every situation it can be in. Returns a list.

    A line is described by four things:
        mine       - how many pieces I have on top in the line (0 to 4)
        theirs     - how many the opponent has on top in the line (0 to 4)
        their_large - 1 if the opponent has a large piece in the line. A large piece can never be
                      covered, so I can never complete this line.
        my_large   - the same for my large pieces, from the opponent's point of view.

    The score is stored at index  mine + 5 * theirs + 25 * their_large + 50 * my_large.
    """
    def one_side(own, other, blocked_by_large):
        """How much a line is worth to the player who has `own` pieces in it."""
        if blocked_by_large or own == 0 or own >= win_length:
            return 0
        return LINE_VALUES[own] * BLOCK_EIGHTHS[other] // 8

    scores = [0] * 100
    for mine in range(5):
        for theirs in range(5):
            for their_large in (0, 1):
                for my_large in (0, 1):
                    index = mine + 5 * theirs + 25 * their_large + 50 * my_large
                    # What the line is worth to me minus what it is worth to the opponent.
                    scores[index] = one_side(mine, theirs, their_large) - one_side(theirs, mine, my_large)
    return scores


@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up the agent. Runs once at the start of each game (we have 30 seconds for this)."""
        super().__init__()
        self.name = "StudentAgent"
        self.allow_moves = bool(constants.ALLOW_MOVES)
        self.win_length = int(constants.WIN_LENGTH)
        self.has_line, self.completing_cells, self.line_masks = build_tables(self.win_length)
        self.line_scores = build_line_scores(self.win_length)

        # Transposition table: remembers positions we already searched, so that when the same
        # position comes up again (by a different order of moves) we can reuse the answer.
        #   position key -> (depth searched, kind of score, score, best move)
        self.table = {}

        # History scores, one list per player, indexed by move code. Every time a move turns out
        # to be good enough to prune the rest, its score goes up. Moves with a high score are
        # tried first next time, and trying good moves first makes alpha-beta prune much more.
        self.history = [[0] * 2048, [0] * 2048]

        self.turns_played = 0               # how many times step() was called in this game

        # Numbers about the last search. Only used to collect the data for the report.
        self.last_depth = 0                 # the deepest search that finished
        self.last_nodes = 0                 # how many positions were looked at
        self.last_stats = (0, 0, 0)         # (table hits, prunings, prunings by the first move tried)
        self.fallbacks = 0                  # how many times the search failed and we used the backup move

    # ------------------------------------------------------------------ the entry point

    def step(self, board, reserves, player, opponent):
        """Return the Move to play. The framework calls this once per turn.

        Takes the same arguments as Agent.step.
        """
        deadline = time.perf_counter() + STEP_TIME_LIMIT * TIME_FRACTION

        # A backup move, in case anything goes wrong below. An illegal move or a crash would
        # cost a strike, so we never want to return without a legal move.
        backup_move = get_valid_moves(board, reserves, player)[0]

        turn_number = self.turns_played
        self.turns_played += 1

        # Python's garbage collector sometimes pauses the program for a moment. That could push
        # a move over the time limit, so we switch it off while searching.
        gc_was_on = gc.isenabled()
        gc.disable()
        try:
            code = self.search(board, reserves, player, turn_number, deadline)
            move = self.code_to_move(code)
            if move is not None and check_move_validity(board, reserves, player, move):
                return move
        except Exception:
            pass                            # fall through to the backup move
        finally:
            if gc_was_on:
                gc.enable()
        self.fallbacks += 1
        return backup_move

    @staticmethod
    def code_to_move(code):
        """Turn a move code (see the top of the file) into a helpers.Move. Returns None for code 0."""
        if not code:
            return None
        size = (code & 3) - 1               # the lowest 2 bits hold size + 1
        dest = (code >> 2) & 15             # the next 4 bits hold the destination cell
        src_plus_one = code >> 6            # the rest holds source cell + 1 (0 = from the reserve)
        dest_pos = (dest // BOARD_SIZE, dest % BOARD_SIZE)
        if src_plus_one:
            src = src_plus_one - 1
            return Move((src // BOARD_SIZE, src % BOARD_SIZE), dest_pos, size)
        return Move(None, dest_pos, size)

    # ------------------------------------------------------------------ the search

    def search(self, board, reserves, player, turn_number, deadline):
        """Search until the deadline. Returns the code of the best move found (0 if none).

        The three helper functions (find_moves, evaluate, negamax) are defined inside this
        function on purpose: that way they can all use the variables below (pieces, reserve, ...)
        directly, which is faster in Python than passing them around or using self.something.
        """
        # ---- step 1: copy the numpy board into the six integers described at the top
        cells = board.tolist()              # plain Python lists are faster to read than numpy
        pieces = [0] * 6                    # pieces[side * 3 + size], side 0 = player 1, side 1 = player 2
        obstacle = 0
        pieces_on_board = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stack = cells[row][col]     # the three slots of this cell: [small, medium, large]
                bit = 1 << (row * BOARD_SIZE + col)
                if stack[0] == OBSTACLE:
                    obstacle |= bit
                    continue
                for size in range(3):
                    owner = stack[size]
                    if owner == PLAYER_1:
                        pieces[size] |= bit
                        pieces_on_board += 1
                    elif owner == PLAYER_2:
                        pieces[3 + size] |= bit
                        pieces_on_board += 1

        # reserve[side * 3 + size] = how many pieces of that size the side still has off the board
        reserve_rows = reserves.tolist()
        reserve = [int(n) for n in reserve_rows[PLAYER_1]] + [int(n) for n in reserve_rows[PLAYER_2]]

        my_side = player - 1                # 0 for player 1, 1 for player 2

        # ---- step 2: work out how many moves are left before the 100-move draw
        can_move_pieces = self.allow_moves
        if can_move_pieces:
            # Each player moves once per turn, so my n-th turn is move number 2n (+1 for player 2).
            moves_played = 2 * turn_number + player - 1
        else:
            # Without board moves every move adds one piece, so we can just count the pieces.
            moves_played = pieces_on_board
        moves_left = max(1, MAX_PLIES - moves_played)
        near_end = moves_left <= NEAR_END_PLIES

        # ---- step 3: get the memory from earlier turns ready
        table = self.table
        if near_end or len(table) > TABLE_MAX_SIZE:
            table.clear()                   # keeps memory small, and old scores are wrong near the end
        history = self.history
        for side_history in history:
            # Shrink the old history scores so that what we learn this turn counts more.
            side_history[:] = [score >> 2 for score in side_history]

        # ---- step 4: local names. Python finds a local variable faster than self.x or a global,
        # and these are used millions of times.
        OPEN_CELLS = ALL_CELLS & ~obstacle  # every cell except the obstacle
        HAS_LINE = self.has_line
        COMPLETING = self.completing_cells
        LINE_SCORES = self.line_scores
        # Lines through the obstacle can never be completed, so the evaluation ignores them.
        usable_lines = [mask for mask in self.line_masks if not mask & obstacle]
        TURN_KEY = 1 << 96                  # the bit of the position key that says whose turn it is
        node_limit = NODE_LIMIT
        clock = time.perf_counter
        table_get = table.get

        nodes = 0                           # positions looked at so far
        best_root_move = 0                  # the best move found so far for the real position
        table_hits = prunings = first_move_prunings = 0     # statistics for the report

        def add_moves(targets, base_code, opp_tops, captures, quiet):
            """Add one move for every cell in `targets` to the lists.

            base_code is the move code without the destination. A move that lands on an opponent's
            top piece goes into `captures`, any other move goes into `quiet`.
            """
            x = targets & opp_tops
            while x:
                bit = x & -x
                x ^= bit
                captures.append(base_code | (bit.bit_length() - 1) << 2)
            x = targets & ~opp_tops
            while x:
                bit = x & -x
                x ^= bit
                quiet.append(base_code | (bit.bit_length() - 1) << 2)

        def find_moves(side, make_lists, include_own_covers):
            """Look at the moves of `side`. Returns (winning_move, captures, quiet, losing_move).

            winning_move - the code of a move that wins immediately, or 0 if there is none.
                           If there is one we return right away and the lists are None.
            captures     - moves that cover an opponent's piece.
            quiet        - all other moves.
                           Both lists are only filled when make_lists is True. (Sometimes we only
                           want to know "can this player win right now?", which is cheaper.)
                           Moves that cover my OWN piece are almost never useful, so they are
                           left out unless include_own_covers is True.
            losing_move  - a move that uncovers a finished line of the opponent (we lose at
                           once), or 0. These moves are never put in the lists.
            """
            mine = side * 3                 # where my three integers start in `pieces`
            theirs = 3 - mine
            my_small = pieces[mine]
            my_medium = pieces[mine + 1]
            my_large = pieces[mine + 2]
            opp_small = pieces[theirs]
            opp_medium = pieces[theirs + 1]
            opp_large = pieces[theirs + 2]
            any_medium = my_medium | opp_medium
            any_small = my_small | opp_small

            # Where can a piece of each size be put down?
            large_spots = OPEN_CELLS & ~(my_large | opp_large)  # anywhere without a large piece
            medium_spots = large_spots & ~any_medium            # ...and without a medium piece
            small_spots = medium_spots & ~any_small             # ...and without a small one: empty cells

            # Which pieces are on top (visible)? A large always is. A medium is on top when
            # there is no large on its cell, and a small when there is no large or medium.
            my_top_mediums = my_medium & large_spots
            my_top_smalls = my_small & medium_spots
            my_tops = my_large | my_top_mediums | my_top_smalls
            opp_tops = opp_large | (opp_medium & large_spots) | (opp_small & medium_spots)

            smalls_left = reserve[mine]
            mediums_left = reserve[mine + 1]
            larges_left = reserve[mine + 2]

            # --- Can I win by placing a new piece? The table gives the cells that finish a line.
            finishing = COMPLETING[my_tops]
            if finishing:
                if larges_left and finishing & large_spots:
                    x = finishing & large_spots
                    return ((x & -x).bit_length() - 1) << 2 | 3, None, None, 0
                if mediums_left and finishing & medium_spots:
                    x = finishing & medium_spots
                    return ((x & -x).bit_length() - 1) << 2 | 2, None, None, 0
                if smalls_left and finishing & small_spots:
                    x = finishing & small_spots
                    return ((x & -x).bit_length() - 1) << 2 | 1, None, None, 0

            # --- Placements from the reserve. The base code is just size + 1 (no source cell).
            captures = []
            quiet = []
            losing_move = 0
            if make_lists:
                if include_own_covers:
                    allowed = ALL_CELLS
                else:
                    allowed = ~my_tops      # skip the cells where my own piece is on top
                if larges_left:
                    add_moves(large_spots & allowed, 3, opp_tops, captures, quiet)
                if mediums_left:
                    add_moves(medium_spots & allowed, 2, opp_tops, captures, quiet)
                if smalls_left:
                    add_moves(small_spots & allowed, 1, opp_tops, captures, quiet)
            if not can_move_pieces:
                return 0, captures, quiet, 0

            # --- Moves of a piece that is already on the board. Only top pieces can be lifted.
            for size_code, movers, spots in ((3, my_large, large_spots),
                                             (2, my_top_mediums, medium_spots),
                                             (1, my_top_smalls, small_spots)):
                x = movers
                while x:
                    src_bit = x & -x
                    x ^= src_bit

                    # Lifting the piece shows what was under it. Work out who is on top of
                    # every cell once this piece is in the air.
                    if size_code == 3 and any_medium & src_bit:
                        something_below = True
                        below_is_mine = my_medium & src_bit
                    elif size_code >= 2 and any_small & src_bit:
                        something_below = True
                        below_is_mine = my_small & src_bit
                    else:
                        something_below = False
                        below_is_mine = 0
                    if not something_below:
                        my_tops_lifted = my_tops ^ src_bit      # the cell becomes empty
                        opp_tops_lifted = opp_tops
                    elif below_is_mine:
                        my_tops_lifted = my_tops                # my other piece shows: still my cell
                        opp_tops_lifted = opp_tops
                    else:
                        my_tops_lifted = my_tops ^ src_bit      # the opponent's piece shows
                        opp_tops_lifted = opp_tops | src_bit

                    # bit_length() of a single cell is its cell number + 1: what the code needs.
                    base_code = src_bit.bit_length() << 6 | size_code

                    if HAS_LINE[opp_tops_lifted]:
                        # Dangerous: lifting this piece uncovers a full line of the opponent.
                        # It is only safe if the piece lands on that line and breaks it again.
                        y = spots
                        while y:
                            bit = y & -y
                            y ^= bit
                            move = base_code | (bit.bit_length() - 1) << 2
                            if HAS_LINE[opp_tops_lifted & ~bit]:
                                losing_move = move              # the line is still there: we lose
                            elif HAS_LINE[my_tops_lifted | bit]:
                                return move, None, None, 0      # line broken and we finish our own
                            elif make_lists:
                                if bit & opp_tops:
                                    captures.append(move)
                                else:
                                    quiet.append(move)
                        continue

                    # Can this piece finish one of my lines by moving?
                    finishing = COMPLETING[my_tops_lifted] & spots
                    if finishing:
                        return base_code | ((finishing & -finishing).bit_length() - 1) << 2, None, None, 0

                    if make_lists:
                        if include_own_covers:
                            targets = spots
                        else:
                            targets = spots & ~my_tops_lifted
                        add_moves(targets, base_code, opp_tops, captures, quiet)
            return 0, captures, quiet, losing_move

        def evaluate(side):
            """Give the position a score from the point of view of `side`, who is about to move.

            Positive means good for `side`. It is only an estimate: it is used when the search
            has to stop before the game is over.
            """
            mine = side * 3
            theirs = 3 - mine
            my_medium = pieces[mine + 1]
            opp_medium = pieces[theirs + 1]
            my_large = pieces[mine + 2]
            opp_large = pieces[theirs + 2]

            # Find the top pieces, the same way as in find_moves.
            no_large = ~(my_large | opp_large)
            no_large_or_medium = no_large & ~(my_medium | opp_medium)
            my_tops = my_large | (my_medium & no_large) | (pieces[mine] & no_large_or_medium)
            opp_tops = opp_large | (opp_medium & no_large) | (pieces[theirs] & no_large_or_medium)

            # 1. Being the one to move, and having more visible pieces than the opponent.
            score = MY_TURN_BONUS + TOP_PIECE_VALUE * (my_tops.bit_count() - opp_tops.bit_count())

            # 2. Every line that can still be completed: count the pieces of both players in it
            #    and look up the score that build_line_scores prepared.
            for mask in usable_lines:
                score += LINE_SCORES[(my_tops & mask).bit_count() + 5 * (opp_tops & mask).bit_count()
                                     + (25 if opp_large & mask else 0) + (50 if my_large & mask else 0)]

            # 3. Buried pieces. An opponent piece under one of mine cannot move or count for a
            #    line while my piece stays there, so the opponent is effectively a piece short.
            their_buried = (pieces[theirs] & (my_medium | my_large)) | (opp_medium & my_large)
            my_buried = (pieces[mine] & (opp_medium | opp_large)) | (my_medium & opp_large)
            score += BURIED_VALUE * (their_buried.bit_count() - my_buried.bit_count())

            # 4. Extra credit when the trapped piece is a medium under my large. In our test
            #    games this made the agent a little stronger than counting all buried pieces the same.
            score += BURIED_MEDIUM_BONUS * ((opp_medium & my_large).bit_count()
                                            - (my_medium & opp_large).bit_count())
            return score

        def negamax(side, depth, alpha, beta, ply, key):
            """Return the score of the current position for `side`, who is about to move.

            side   - 0 or 1, the player to move.
            depth  - how many more moves deep to search. At 0 we stop and evaluate.
            alpha  - the best score `side` is already sure of from earlier in the search.
            beta   - the best score the opponent is already sure of. If we find a score of beta
                     or more, the opponent will never let us get here, so we can stop (prune).
            ply    - how many moves we are away from the real position (0 = the real position).
            key    - one big integer that identifies this exact position, for the table.
            """
            nonlocal nodes, best_root_move, table_hits, prunings, first_move_prunings

            # Check the clock, but only every 1024 positions because reading the clock is slow.
            nodes += 1
            if nodes % 1024 == 0 and (clock() > deadline or (node_limit and nodes >= node_limit)):
                raise OutOfTime

            # The 100-move limit: the game is a draw. A draw is scored slightly bad for us.
            if ply >= moves_left:
                return -DRAW_PENALTY if side == my_side else DRAW_PENALTY

            # ---- 1. Have we searched this position before? Then maybe we can reuse the result.
            table_move = 0
            if depth > 0:
                # Near the end of the game the same position is worth something different
                # depending on how many moves are left, so the move number becomes part of the key.
                table_key = key | (ply << 100) if near_end else key
                entry = table_get(table_key)
                if entry is not None:
                    saved_depth, saved_kind, saved_score, table_move = entry
                    # Only trust the saved score if it came from a search at least this deep.
                    if saved_depth >= depth and ply:
                        # Win scores are saved as "moves from that position"; turn them back
                        # into "moves from the real position".
                        if saved_score > WIN_ZONE:
                            saved_score -= ply
                        elif saved_score < -WIN_ZONE:
                            saved_score += ply
                        if saved_kind == EXACT:
                            table_hits += 1
                            return saved_score
                        if saved_kind == AT_LEAST:
                            if saved_score >= beta:
                                table_hits += 1
                                return saved_score
                        elif saved_score <= alpha:          # AT_MOST
                            table_hits += 1
                            return saved_score
                winning_move, captures, quiet, losing_move = find_moves(side, True, False)
            else:
                # At depth 0 we first only ask "can I win right now?" (no lists needed).
                winning_move, captures, quiet, losing_move = find_moves(side, False, False)

            # ---- 2. If I can win immediately, that is the answer.
            if winning_move:
                if ply == 0:
                    best_root_move = winning_move
                return WIN_SCORE - ply - 1

            # ---- 3. Out of depth: stop and evaluate, unless the opponent threatens to win on
            # their next move. An evaluation in the middle of a threat would be misleading, so
            # in that case we keep searching a little deeper (at most THREAT_EXTRA_DEPTH moves).
            if depth <= 0:
                if depth <= -THREAT_EXTRA_DEPTH or not find_moves(1 - side, False, False)[0]:
                    return evaluate(side)           # no threat (or deep enough already)
                winning_move, captures, quiet, losing_move = find_moves(side, True, False)

            # ---- 4. Rare case: no moves in the lists.
            if not captures and not quiet:
                # Try again, this time also allowing moves that cover my own pieces.
                winning_move, captures, quiet, losing_move = find_moves(side, True, True)
                if not captures and not quiet:
                    if losing_move:                 # every move uncovers the opponent's line
                        if ply == 0:
                            best_root_move = losing_move
                        return -(WIN_SCORE - ply - 1)
                    if can_move_pieces:             # no legal move at all: the rules say I lose
                        return -(WIN_SCORE - ply)
                    # Without board moves, having no move just passes the turn.
                    return -negamax(1 - side, depth, -beta, -alpha, ply + 1, key ^ TURN_KEY)

            # ---- 5. Put the moves in a good order. Alpha-beta prunes the most when the best
            # move is tried first. Order: the best move from the table, then captures, then
            # quiet moves, each group sorted by its history score.
            other = 1 - side
            side_history = history[side]
            history_score = side_history.__getitem__    # history_score(move) -> its score
            if len(captures) > 1:
                captures.sort(key=history_score, reverse=True)
            quiet.sort(key=history_score, reverse=True)
            late_from = len(captures) + LATE_MOVE_START  # moves from this position on are "late"
            moves = captures + quiet
            if table_move and table_move in moves:
                moves.remove(table_move)
                moves.insert(0, table_move)
                late_from += 1

            # ---- 6. Try every move.
            mine = side * 3
            original_alpha = alpha
            best_score = -INFINITY
            best_move = 0
            for i, move in enumerate(moves):
                # Play the move on the bitboards.
                piece_index = mine + (move & 3) - 1     # which of the six integers changes
                dest_bit = 1 << ((move >> 2) & 15)
                src_plus_one = move >> 6
                if src_plus_one:
                    # A board move: switch off the source cell and switch on the destination.
                    changed = dest_bit | 1 << (src_plus_one - 1)
                    pieces[piece_index] ^= changed
                else:
                    # A placement: switch on the destination and take the piece from the reserve.
                    changed = dest_bit
                    pieces[piece_index] ^= changed
                    reserve[piece_index] -= 1
                # The key of the new position: flip the same bits in the key, and flip the turn.
                new_key = key ^ (changed << (piece_index * 16)) ^ TURN_KEY

                # Search the position after the move. The score comes back from the opponent's
                # point of view, so we put a minus sign in front (and swap and negate alpha/beta).
                if i == 0:
                    # The first move is probably the best one: search it fully.
                    score = -negamax(other, depth - 1, -beta, -alpha, ply + 1, new_key)
                else:
                    # For the other moves we only want to know "is it better than alpha?".
                    # Asking that yes/no question (a window of width 1) is much cheaper than
                    # asking for the exact score. Late quiet moves are also searched one move
                    # shallower, because they are rarely the best.
                    if depth >= 3 and i >= late_from:
                        reduction = 1
                    else:
                        reduction = 0
                    score = -negamax(other, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, new_key)
                    # If the answer is "yes, it is better", we need the real score: search again
                    # properly.
                    if score > alpha and (reduction or score < beta):
                        score = -negamax(other, depth - 1, -beta, -alpha, ply + 1, new_key)

                # Take the move back: flipping the same bits again undoes it.
                pieces[piece_index] ^= changed
                if not src_plus_one:
                    reserve[piece_index] += 1

                if score > best_score:
                    best_score = score
                    best_move = move
                    if score > alpha:
                        alpha = score
                        if ply == 0:
                            best_root_move = move
                        if score >= beta:
                            # Too good: the opponent will avoid this position, so the remaining
                            # moves do not matter. This is the alpha-beta pruning.
                            prunings += 1
                            if i == 0:
                                first_move_prunings += 1
                            if depth > 0:
                                side_history[move] += depth * depth     # remember it was a good move
                            break

            # ---- 7. Save the result in the table for next time.
            if depth > 0:
                if best_score >= beta:
                    kind = AT_LEAST         # we pruned, so the true score could be even higher
                elif best_score > original_alpha:
                    kind = EXACT
                else:
                    kind = AT_MOST          # no move reached alpha, so this is only an upper limit
                score_to_save = best_score
                if score_to_save > WIN_ZONE:
                    score_to_save += ply
                elif score_to_save < -WIN_ZONE:
                    score_to_save -= ply
                table[table_key] = (depth, kind, score_to_save, best_move)
            return best_score

        # ---- step 5: iterative deepening. Search depth 1, then 2, then 3, ... Each search is
        # quick compared to the next one, and it fills the table and history scores, which makes
        # the deeper search much faster. When the time is up, negamax raises OutOfTime and we
        # keep the best move found so far.
        key = my_side << 96                 # build the key of the real position
        for i in range(6):
            key |= pieces[i] << (i * 16)
        self.last_depth = 0
        try:
            for depth in range(1, moves_left + 1):
                score = negamax(my_side, depth, -INFINITY, INFINITY, 0, key)
                self.last_depth = depth
                if score > WIN_ZONE or score < -WIN_ZONE:
                    break                   # a forced win or loss was found: deeper search cannot change it
        except OutOfTime:
            pass
        self.last_nodes = nodes
        self.last_stats = (table_hits, prunings, first_move_prunings)
        return best_root_move
