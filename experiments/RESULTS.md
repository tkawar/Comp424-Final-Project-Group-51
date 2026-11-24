# Measurements for the report

Produced by `experiments/collect.py` and `experiments/summarize.py`. The student agent plays every game
on the 2-second clock; each seed (obstacle and removed size) is played twice with the seats swapped.

## Results

| opponent | games | W | D | L | score | mean plies | late moves | fallback moves |
|---|---|---|---|---|---|---|---|---|
| experiments/baseline_ab.py | 40 | 35 | 5 | 0 | 93.8% | 48 | 0 | 0 |
| greedy | 60 | 59 | 1 | 0 | 99.2% | 45 | 0 | 0 |
| self | 24 | 0 | 24 | 0 | 50.0% | 100 | 0 | 0 |

### Results by seat

| seat | opponent | games | W | D | L | score |
|---|---|---|---|---|---|---|
| first (P1) | experiments/baseline_ab.py | 20 | 17 | 3 | 0 | 92.5% |
| first (P1) | greedy | 30 | 29 | 1 | 0 | 98.3% |
| first (P1) | self | 12 | 0 | 12 | 0 | 50.0% |
| second (P2) | experiments/baseline_ab.py | 20 | 18 | 2 | 0 | 95.0% |
| second (P2) | greedy | 30 | 30 | 0 | 0 | 100.0% |
| second (P2) | self | 12 | 0 | 12 | 0 | 50.0% |

### Results by obstacle position

| obstacle position | opponent | games | W | D | L | score |
|---|---|---|---|---|---|---|
| corner | experiments/baseline_ab.py | 8 | 7 | 1 | 0 | 93.8% |
| corner | greedy | 10 | 10 | 0 | 0 | 100.0% |
| corner | self | 4 | 0 | 4 | 0 | 50.0% |
| edge | experiments/baseline_ab.py | 24 | 21 | 3 | 0 | 93.8% |
| edge | greedy | 40 | 39 | 1 | 0 | 98.8% |
| edge | self | 18 | 0 | 18 | 0 | 50.0% |
| centre | experiments/baseline_ab.py | 8 | 7 | 1 | 0 | 93.8% |
| centre | greedy | 10 | 10 | 0 | 0 | 100.0% |
| centre | self | 2 | 0 | 2 | 0 | 50.0% |

### Results by removed size

| removed size | opponent | games | W | D | L | score |
|---|---|---|---|---|---|---|
| S | experiments/baseline_ab.py | 12 | 9 | 3 | 0 | 87.5% |
| S | greedy | 16 | 16 | 0 | 0 | 100.0% |
| S | self | 6 | 0 | 6 | 0 | 50.0% |
| M | experiments/baseline_ab.py | 6 | 6 | 0 | 0 | 100.0% |
| M | greedy | 12 | 11 | 1 | 0 | 95.8% |
| M | self | 6 | 0 | 6 | 0 | 50.0% |
| L | experiments/baseline_ab.py | 22 | 20 | 2 | 0 | 95.5% |
| L | greedy | 32 | 32 | 0 | 0 | 100.0% |
| L | self | 12 | 0 | 12 | 0 | 50.0% |

## Search by phase of the game

Every move the student agent searched, over all games. Branching is the number of legal moves at the root;
the effective branching factor is nodes^(1/depth); a full minimax tree of the same depth would have
branching^depth nodes.

| plies | moves | legal moves | depth | nodes | effective branching | minimax nodes / searched nodes | table cutoffs per 1000 nodes | cutoffs on first move |
|---|---|---|---|---|---|---|---|---|
| 0-7 | 485 | 48 | 6.8 | 474,079 | 6.9 | 10^5.8 | 56 | 89% |
| 8-15 | 476 | 46 | 6.9 | 499,356 | 6.9 | 10^5.7 | 49 | 84% |
| 16-31 | 841 | 48 | 7.1 | 475,635 | 6.5 | 10^6.2 | 55 | 78% |
| 32-63 | 835 | 47 | 7.4 | 463,759 | 6.0 | 10^6.7 | 68 | 79% |
| 64-99 | 528 | 45 | 6.8 | 413,515 | 6.9 | 10^5.8 | 59 | 84% |

### Depth by obstacle position and removed size

| group | moves | legal moves | depth |
|---|---|---|---|
| obstacle centre | 402 | 47 | 7.1 |
| obstacle corner | 548 | 47 | 7.0 |
| obstacle edge | 2215 | 47 | 7.1 |
| removed L | 1494 | 46 | 7.0 |
| removed M | 701 | 47 | 7.0 |
| removed S | 970 | 48 | 7.1 |

## Timing

- moves timed: 3534
- mean 1.304 s, median 1.501 s, 99th percentile 1.505 s, max 1.523 s
- moves over the 2.0 s limit: 0
- moves answered at once (a forced win or loss was found): 369
- board moves (as opposed to placements): 76% of moves
