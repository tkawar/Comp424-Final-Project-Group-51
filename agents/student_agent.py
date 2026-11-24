"""Student agent: edit ONLY this file.

Iterative-deepening alpha-beta (negamax) on a bitboard copy of the position.

The position is kept as six 16-bit masks, one per (player, size), bit r * 4 + c for cell (r, c).
From them the search derives the top piece of every cell with a few bit operations, tests a win
with one table lookup, and plays and takes back a move by flipping bits. On top of that:
  - iterative deepening with a wall-clock deadline (always a finished depth to fall back on),
  - a transposition table keyed on the exact position,
  - principal-variation search with late-move reductions and history move ordering,
  - win-in-one detection at every node, and an extension while a side is under a direct threat,
  - an evaluation that scores every line that can still be completed.
"""
from agents.agent import Agent
from store import register_agent
from helpers import Move, get_valid_moves, check_move_validity, LINES
import constants
from constants import BOARD_SIZE, PLAYER_1, PLAYER_2, OBSTACLE, MAX_PLIES, STEP_TIME_LIMIT
import gc
import time

_N = BOARD_SIZE * BOARD_SIZE          # 16 cells
_FULL = (1 << _N) - 1

_WIN = 1_000_000                      # score of a win at the root; a win n plies away is _WIN - n
_MATE = _WIN - 1000                   # any score beyond this is a proven win or loss
_INF = 10 * _WIN

_TIME_FRACTION = 0.82                 # share of STEP_TIME_LIMIT the search may use
_TT_MAX = 300_000                     # transposition table entries kept between moves
_EXT_LIMIT = 4                        # extra plies searched while the side to move is under threat
_LMR_FROM = 4                         # quiet moves after this many are searched one ply shallower
_CONTEMPT = 10                        # a draw is worth this much less than an even position
_VISIBLE = 3                          # evaluation: value of one more piece on top than the opponent
_TEMPO = 3                            # evaluation: value of being the side to move
_NEAR_END = 28                        # plies before the cap from which positions depend on the ply
_LINE_BASE = (1, 4, 16)                # evaluation: value of 1, 2, 3, ... own top pieces in a line
_BLOCK = (8, 4, 3, 2, 2)              # evaluation: eighths of that value left with 0, 1, ... opposing pieces
_LINE_COUNT = 0                       # evaluation: value of each line a top piece stands on
_VIS_SIZE = (0, 0, 0)                 # evaluation: extra value of a small / medium / large piece on top
_RESERVE = (0, 0, 0)                  # evaluation: value of a small / medium / large piece in reserve
_THREAT = 0                           # evaluation: value of a cell that would complete a line
_BURIED = 8                           # evaluation: value of an opposing piece buried under one's own
_KILLERS = 0                          # 1: try the quiet moves that cut off at this ply first
_NULL_R = 0                           # null-move pruning reduction (0 = off)
_FUT = 0                              # futility margin at depth 1 (0 = off)
_LMR2 = 0                             # quiet moves this far after _LMR_FROM lose two plies (0 = off)
_CHECK_EXT = 0                        # 1: search one ply deeper when the side to move is under threat
_REP = 0                              # penalty for a root move into a position already played
_HIST_INIT = 0                        # initial move-ordering value of each line through the target cell
_NODE_LIMIT = 0                       # testing only: stop after this many positions (0 = use the clock)

_EXACT, _LOWER, _UPPER = 0, 1, 2

_TABLES = {}                          # win_length -> (win table, completion table, line masks)


class _Timeout(Exception):
    """Raised inside the search when the deadline has passed."""


def _tables(win_length):
    """Return (win, complete, masks) for this win length, built once per process.

    win[t]      - 1 if the 16-bit set of cells t contains a whole line.
    complete[t] - the cells c for which t plus c contains a whole line (for t without a line).
    masks       - the bit mask of every line.
    """
    if win_length in _TABLES:
        return _TABLES[win_length]
    masks = []
    for line in LINES[win_length]:
        mk = 0
        for r, c in line:
            mk |= 1 << (r * BOARD_SIZE + c)
        masks.append(mk)
    win = bytearray(1 << _N)
    complete = [0] * (1 << _N)
    for mk in masks:
        rest = _FULL ^ mk
        sub = rest
        while True:                                   # every superset of the line
            win[mk | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
        x = mk
        while x:
            cb = x & -x
            x ^= cb
            base = mk ^ cb                            # the line without one cell
            rest = _FULL ^ mk
            sub = rest
            while True:
                complete[base | sub] |= cb
                if sub == 0:
                    break
                sub = (sub - 1) & rest
    _TABLES[win_length] = (win, complete, masks)
    return _TABLES[win_length]


def _line_table(win_length):
    """Return the evaluation of one line, indexed by a + 5 * b + 25 * opp_large + 50 * my_large.

    a, b      - the number of top pieces in the line of the side to evaluate and of the opponent.
    opp_large - the opponent has a large piece on top in the line (it cannot be covered).
    my_large  - the same for the side to evaluate.
    """
    base = ([0] + list(_LINE_BASE)[:win_length - 1] + [0] * 5)[:5]

    def value(own, other, blocked):
        if blocked or own == 0 or own >= win_length:
            return 0
        return base[own] * _BLOCK[other] // 8         # every piece of the other side must be covered

    table = [0] * 100
    for a in range(5):
        for b in range(5):
            for ol in (0, 1):
                for ml in (0, 1):
                    table[a + 5 * b + 25 * ol + 50 * ml] = value(a, b, ol) - value(b, a, ml) + _LINE_COUNT * (a - b)
    return table


@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up the agent: build the lookup tables. Runs once per game."""
        super().__init__()
        self.name = "StudentAgent"
        self.allow_moves = bool(constants.ALLOW_MOVES)
        self.win_length = int(constants.WIN_LENGTH)
        self.win, self.complete, self.masks = _tables(self.win_length)
        self.line_table = _line_table(self.win_length)
        self.tt = {}                                  # position -> (depth, flag, score, move)
        nlines = [sum(1 for mk in self.masks if mk >> c & 1) for c in range(_N)]
        self.prior = [_HIST_INIT * nlines[(m >> 2) & 15] for m in range(2048)]
        self.hist = [list(self.prior), list(self.prior)]   # history score of every move, per side
        self.seen = {}                                # positions left after my moves -> times played
        self.calls = 0                                # number of step() calls so far
        self.last_depth = 0                           # depth finished by the last search (for tests)
        self.last_nodes = 0

    # ------------------------------------------------------------------ the entry point

    def step(self, board, reserves, player, opponent):
        """Return the Move to play: the best move found before the deadline.

        Takes the same arguments as Agent.step.
        """
        start = time.perf_counter()
        deadline = start + STEP_TIME_LIMIT * _TIME_FRACTION
        fallback = get_valid_moves(board, reserves, player)[0]
        k = self.calls
        self.calls += 1
        was_enabled = gc.isenabled()
        gc.disable()                                  # the search makes no reference cycles
        try:
            code = self._search(board, reserves, player, k, deadline)
            move = self._to_move(code)
            if move is not None and check_move_validity(board, reserves, player, move):
                return move
        except Exception:
            pass
        finally:
            if was_enabled:
                gc.enable()
        return fallback

    @staticmethod
    def _to_move(code):
        """Return the helpers.Move of a move code, or None for code 0.

        A code is (size + 1) | dest << 2 | (src + 1) << 6, with src + 1 == 0 for a placement.
        """
        if not code:
            return None
        size = (code & 3) - 1
        dest = (code >> 2) & 15
        src = code >> 6
        d = (dest // BOARD_SIZE, dest % BOARD_SIZE)
        if src:
            return Move(((src - 1) // BOARD_SIZE, (src - 1) % BOARD_SIZE), d, size)
        return Move(None, d, size)

    # ------------------------------------------------------------------ the search

    def _search(self, board, reserves, player, k, deadline):
        """Search the position until the deadline and return the code of the best move found."""
        cells = board.tolist()
        bb = [0] * 6                                  # bb[p * 3 + s]: the cells with a size-s piece of side p
        obstacle = 0
        on_board = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stack = cells[r][c]
                bit = 1 << (r * BOARD_SIZE + c)
                if stack[0] == OBSTACLE:
                    obstacle |= bit
                    continue
                for s in range(3):
                    v = stack[s]
                    if v == PLAYER_1:
                        bb[s] |= bit
                        on_board += 1
                    elif v == PLAYER_2:
                        bb[3 + s] |= bit
                        on_board += 1
        rl = reserves.tolist()
        res = [int(x) for x in rl[PLAYER_1]] + [int(x) for x in rl[PLAYER_2]]
        root_side = player - 1

        ALLOW = self.allow_moves
        # With board moves every turn is one call, so the k-th call is ply 2k + player - 1.
        # Without them every ply places one piece (a rare pass aside).
        root_ply = 2 * k + player - 1 if ALLOW else on_board
        plies_left = max(1, MAX_PLIES - root_ply)
        near_end = plies_left <= _NEAR_END

        tt = self.tt
        if near_end or len(tt) > _TT_MAX:
            tt.clear()
        hist = self.hist
        prior = self.prior
        for h in hist:
            for i, v in enumerate(h):
                if v != prior[i]:
                    h[i] = prior[i] + ((v - prior[i]) >> 2)
        seen = self.seen
        VS0, VS1, VS2 = _VIS_SIZE
        RS0, RS1, RS2 = _RESERVE
        THREAT = _THREAT
        BURIED = _BURIED
        EXTRA = bool(VS0 or VS1 or VS2 or RS0 or RS1 or RS2 or THREAT or BURIED)
        KILLERS = _KILLERS
        NULL_R = _NULL_R
        FUT = _FUT
        LMR2 = _LMR2
        CHECK_EXT = _CHECK_EXT
        REP = _REP
        killers = [[0, 0] for _ in range(MAX_PLIES + 64)]
        cur_depth = 1

        FREE = _FULL & ~obstacle
        WINT = self.win
        COMP = self.complete
        LT = self.line_table
        live = [mk for mk in self.masks if not mk & obstacle]
        SIDEKEY = 1 << 96
        TEMPO = _TEMPO
        VISIBLE = _VISIBLE
        EXT_LIMIT = _EXT_LIMIT
        LMR_FROM = _LMR_FROM
        NODE_LIMIT = _NODE_LIMIT
        perf = time.perf_counter
        tt_get = tt.get
        nodes = 0
        root_best = 0

        def gen(side, build, full):
            """Look at the moves of `side`. Return (win, captures, quiet, loser).

            win      - the code of a move that wins at once, or 0. When there is one, the lists
                       are not returned.
            captures - moves onto an opponent's piece; quiet - the other moves (both only when
                       `build`). Moves onto the side's own pieces are left out unless `full`.
            loser    - a move that uncovers an opponent's line and so loses at once, or 0; such
                       moves are never in the lists.
            """
            mi = side * 3
            oi = 3 - mi
            mS = bb[mi]; mM = bb[mi + 1]; mL = bb[mi + 2]
            oS = bb[oi]; oM = bb[oi + 1]; oL = bb[oi + 2]
            occM = mM | oM
            occS = mS | oS
            dL = FREE & ~(mL | oL)                    # cells a large / medium / small can go to
            dM = dL & ~occM
            dS = dM & ~occS
            mMt = mM & dL                             # my mediums and smalls that are on top
            mSt = mS & dM
            mtop = mL | mMt | mSt
            otop = oL | (oM & dL) | (oS & dM)
            rS = res[mi]; rM = res[mi + 1]; rL = res[mi + 2]

            c = COMP[mtop]
            if c:                                     # a placement that completes a line
                if rL and c & dL:
                    x = c & dL
                    return ((x & -x).bit_length() - 1) << 2 | 3, None, None, 0
                if rM and c & dM:
                    x = c & dM
                    return ((x & -x).bit_length() - 1) << 2 | 2, None, None, 0
                if rS and c & dS:
                    x = c & dS
                    return ((x & -x).bit_length() - 1) << 2 | 1, None, None, 0

            caps = []
            quiet = []
            loser = 0
            if build:
                keep = _FULL if full else ~mtop
                if rL:
                    d = dL & keep
                    x = d & otop
                    while x:
                        b = x & -x; x ^= b
                        caps.append((b.bit_length() - 1) << 2 | 3)
                    x = d & ~otop
                    while x:
                        b = x & -x; x ^= b
                        quiet.append((b.bit_length() - 1) << 2 | 3)
                if rM:
                    d = dM & keep
                    x = d & otop
                    while x:
                        b = x & -x; x ^= b
                        caps.append((b.bit_length() - 1) << 2 | 2)
                    x = d & ~otop
                    while x:
                        b = x & -x; x ^= b
                        quiet.append((b.bit_length() - 1) << 2 | 2)
                if rS:
                    x = dS & keep                     # a small only goes to an empty cell
                    while x:
                        b = x & -x; x ^= b
                        quiet.append((b.bit_length() - 1) << 2 | 1)
            if not ALLOW:
                return 0, caps, quiet, 0

            # Board moves. For every piece that can be lifted: (its bit, my tops and the
            # opponent's tops once it is lifted, where it can go, its size code).
            srcs = []
            x = mL
            while x:
                sb = x & -x; x ^= sb
                if occM & sb:
                    if mM & sb:
                        srcs.append((sb, mtop, otop, dL, 3))
                    else:
                        srcs.append((sb, mtop ^ sb, otop | sb, dL, 3))
                elif occS & sb:
                    if mS & sb:
                        srcs.append((sb, mtop, otop, dL, 3))
                    else:
                        srcs.append((sb, mtop ^ sb, otop | sb, dL, 3))
                else:
                    srcs.append((sb, mtop ^ sb, otop, dL, 3))
            x = mMt
            while x:
                sb = x & -x; x ^= sb
                if occS & sb:
                    if mS & sb:
                        srcs.append((sb, mtop, otop, dM, 2))
                    else:
                        srcs.append((sb, mtop ^ sb, otop | sb, dM, 2))
                else:
                    srcs.append((sb, mtop ^ sb, otop, dM, 2))
            x = mSt
            while x:
                sb = x & -x; x ^= sb
                srcs.append((sb, mtop ^ sb, otop, dS, 1))

            for sb, mb, ob, d, code in srcs:
                base = sb.bit_length() << 6 | code
                if WINT[ob]:                          # lifting uncovers an opponent's line
                    x = d
                    while x:
                        b = x & -x; x ^= b
                        m = base | (b.bit_length() - 1) << 2
                        if WINT[ob & ~b]:
                            loser = m                 # the line is still there: the opponent wins
                        elif WINT[mb | b]:
                            return m, None, None, 0
                        elif build:
                            if b & otop:
                                caps.append(m)
                            else:
                                quiet.append(m)
                    continue
                c = COMP[mb] & d
                if c:
                    return base | ((c & -c).bit_length() - 1) << 2, None, None, 0
                if build:
                    if not full:
                        d &= ~mb
                    x = d & otop
                    while x:
                        b = x & -x; x ^= b
                        caps.append(base | (b.bit_length() - 1) << 2)
                    x = d & ~otop
                    while x:
                        b = x & -x; x ^= b
                        quiet.append(base | (b.bit_length() - 1) << 2)
            return 0, caps, quiet, loser

        def evaluate(side):
            """Return the value of the position for `side`, the side to move."""
            mi = side * 3
            oi = 3 - mi
            mM = bb[mi + 1]; oM = bb[oi + 1]
            mL = bb[mi + 2]; oL = bb[oi + 2]
            nl = ~(mL | oL)
            nm = nl & ~(mM | oM)
            mMt = mM & nl; oMt = oM & nl
            mSt = bb[mi] & nm; oSt = bb[oi] & nm
            mtop = mL | mMt | mSt
            otop = oL | oMt | oSt
            sc = TEMPO + VISIBLE * (mtop.bit_count() - otop.bit_count())
            for mk in live:
                sc += LT[(mtop & mk).bit_count() + 5 * (otop & mk).bit_count()
                         + (25 if oL & mk else 0) + (50 if mL & mk else 0)]
            if EXTRA:
                if VS0 or VS1 or VS2:
                    sc += (VS0 * (mSt.bit_count() - oSt.bit_count()) + VS1 * (mMt.bit_count() - oMt.bit_count())
                           + VS2 * (mL.bit_count() - oL.bit_count()))
                if RS0 or RS1 or RS2:
                    sc += (RS0 * (res[mi] - res[oi]) + RS1 * (res[mi + 1] - res[oi + 1])
                           + RS2 * (res[mi + 2] - res[oi + 2]))
                if THREAT:
                    sc += THREAT * ((COMP[mtop] & FREE & ~oL).bit_count() - (COMP[otop] & FREE & ~mL).bit_count())
                if BURIED:
                    sc += BURIED * (((bb[oi] & (mM | mL)) | (oM & mL)).bit_count()
                                    - ((bb[mi] & (oM | oL)) | (mM & oL)).bit_count())
            return sc

        def negamax(side, depth, alpha, beta, ply, key):
            """Return the value for `side`, to move in a position where nobody has a line."""
            nonlocal nodes, root_best
            nodes += 1
            if not nodes & 1023 and (perf() > deadline or (NODE_LIMIT and nodes >= NODE_LIMIT)):
                raise _Timeout
            if ply >= plies_left:                     # the ply cap: a draw
                return -_CONTEMPT if side == root_side else _CONTEMPT

            ttm = 0
            if depth > 0:
                tk = key | (ply << 100) if near_end else key
                e = tt_get(tk)
                if e is not None:
                    ed, ef, ev, ttm = e
                    if ed >= depth and ply:
                        if ev > _MATE:
                            ev -= ply
                        elif ev < -_MATE:
                            ev += ply
                        if ef == _EXACT:
                            return ev
                        if ef == _LOWER:
                            if ev >= beta:
                                return ev
                        elif ev <= alpha:
                            return ev
                win, caps, quiet, loser = gen(side, True, False)
            else:
                win, caps, quiet, loser = gen(side, False, False)
            if win:
                if ply == 0:
                    root_best = win
                return _WIN - ply - 1

            if depth <= 0:
                # Quiet unless the opponent threatens to win at once: then look at the replies.
                if depth <= -EXT_LIMIT or not gen(1 - side, False, False)[0]:
                    return evaluate(side)
                win, caps, quiet, loser = gen(side, True, False)

            if not caps and not quiet:
                win, caps, quiet, loser = gen(side, True, True)      # also moves onto own pieces
                if not caps and not quiet:
                    if loser:                         # every move uncovers the opponent's line
                        if ply == 0:
                            root_best = loser
                        return -(_WIN - ply - 1)
                    if ALLOW:                         # no legal move: a loss
                        return -(_WIN - ply)
                    return -negamax(1 - side, depth, -beta, -alpha, ply + 1, key ^ SIDEKEY)   # a pass

            other = 1 - side
            best = -_INF
            if depth > 0 and ply:
                if CHECK_EXT and ply < 2 * cur_depth + 2 and gen(other, False, False)[0]:
                    depth += 1                        # under threat: the replies are forced
                elif NULL_R and depth >= 3 and beta - alpha == 1 and -_MATE < beta < _MATE:
                    v = -negamax(other, depth - 1 - NULL_R, -beta, 1 - beta, ply + 1, key ^ SIDEKEY)
                    if beta <= v < _MATE:
                        return v
                if FUT and depth == 1:
                    se = evaluate(side) + FUT
                    if se <= alpha:                   # quiet moves will not reach alpha
                        quiet = []
                        best = se

            hs = hist[side]
            hget = hs.__getitem__
            if len(caps) > 1:
                caps.sort(key=hget, reverse=True)
            quiet.sort(key=hget, reverse=True)
            if KILLERS:
                kl = killers[ply]
                for km in (kl[1], kl[0]):
                    if km and km in quiet:
                        quiet.remove(km)
                        quiet.insert(0, km)
            late = len(caps) + LMR_FROM
            moves = caps + quiet
            if ttm:
                try:
                    moves.remove(ttm)
                except ValueError:
                    pass
                else:
                    moves.insert(0, ttm)
                    late += 1

            mi = side * 3
            alpha0 = alpha
            best_move = 0
            i = 0
            pen = 0
            for m in moves:
                idx = mi + (m & 3) - 1
                db = 1 << ((m >> 2) & 15)
                src = m >> 6
                if src:
                    flip = db | 1 << (src - 1)
                    bb[idx] ^= flip
                else:
                    flip = db
                    bb[idx] ^= db
                    res[idx] -= 1
                k2 = key ^ (flip << (idx * 16)) ^ SIDEKEY

                if not ply and REP:
                    pen = REP * seen.get(k2, 0)       # do not walk back into a position already played
                if i == 0:
                    v = -negamax(other, depth - 1, -beta - pen, -alpha - pen, ply + 1, k2) - pen
                else:
                    if depth >= 3 and i >= late:
                        r = 2 if LMR2 and depth >= 5 and i >= late + LMR2 else 1
                    else:
                        r = 0
                    v = -negamax(other, depth - 1 - r, -alpha - 1 - pen, -alpha - pen, ply + 1, k2) - pen
                    if v > alpha and (r or v < beta):
                        v = -negamax(other, depth - 1, -beta - pen, -alpha - pen, ply + 1, k2) - pen

                bb[idx] ^= flip
                if not src:
                    res[idx] += 1

                if v > best:
                    best = v
                    best_move = m
                    if v > alpha:
                        alpha = v
                        if ply == 0:
                            root_best = m
                        if v >= beta:
                            if depth > 0:
                                hs[m] += depth * depth
                                if KILLERS and m in quiet:
                                    kl = killers[ply]
                                    if kl[0] != m:
                                        kl[1] = kl[0]
                                        kl[0] = m
                            break
                i += 1

            if depth > 0:
                if best >= beta:
                    flag = _LOWER
                elif best > alpha0:
                    flag = _EXACT
                else:
                    flag = _UPPER
                sv = best
                if sv > _MATE:
                    sv += ply
                elif sv < -_MATE:
                    sv -= ply
                tt[tk] = (depth, flag, sv, best_move)
            return best

        key = root_side << 96
        for i in range(6):
            key |= bb[i] << (i * 16)
        self.last_depth = 0
        try:
            for depth in range(1, plies_left + 1):
                cur_depth = depth
                v = negamax(root_side, depth, -_INF, _INF, 0, key)
                self.last_depth = depth
                if v > _MATE or v < -_MATE:           # a forced win or loss: nothing more to find
                    break
        except _Timeout:
            pass
        self.last_nodes = nodes
        if root_best:                                 # remember the position this move leaves
            idx = root_side * 3 + (root_best & 3) - 1
            flip = 1 << ((root_best >> 2) & 15)
            if root_best >> 6:
                flip |= 1 << ((root_best >> 6) - 1)
            k2 = key ^ (flip << (idx * 16)) ^ SIDEKEY
            seen[k2] = seen.get(k2, 0) + 1
        return root_best
