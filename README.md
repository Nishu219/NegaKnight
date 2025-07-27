# NegaKnightV7 - Enhanced Positional Chess Engine

NegaKnightV7 is a chess engine written in Python, leveraging the `python-chess` library. It implements an enhanced Negamax algorithm with several advanced features to improve its playing strength, including:

- **Zobrist Hashing** for transposition table lookups
- **Transposition Table** to store and retrieve previously computed positions
- **Iterative Deepening** for a more robust search
- **Quiescence Search** to evaluate noisy positions accurately
- **Alpha-Beta Pruning** to efficiently trim the search tree
- **Null Move Pruning** for further search space reduction
- **Late Move Reductions** to prioritize important moves
- **Aspiration Windows** to refine the search bounds
- **Killer Moves** and **History Heuristic** for effective move ordering
- **Advanced Positional Evaluation** including:
    - Piece-Square Tables (PST) for midgame and endgame
    - See (Static Exchange Evaluation) for tactical captures
    - Mobility Evaluation
    - King Safety and Attack Evaluation
    - Pawn Structure (passed, doubled, isolated pawns)
    - Outpost and Rook on Open/Semi-Open File bonuses
    - Minor Piece Activity and Center Control

---

## Features

* **Strong Chess AI**: Utilizes a combination of search algorithms and a detailed evaluation function to play competitive chess.
* **UCI/SAN Input**: Supports both Universal Chess Interface (UCI) and Standard Algebraic Notation (SAN) for human moves.
* **Game Recording**: Saves games in PGN (Portable Game Notation) format.
* **Configurable Parameters**: Easily adjust search depth, time limits, and other engine parameters.

---

## Getting Started

### Prerequisites

Before running NegaKnightV7, you need to have Python 3 and the `python-chess` library installed.

You can adjust several parameters at the top of the your_engine_file_name.py file to fine-tune the engine's behavior:

MAX_DEPTH: The maximum depth the iterative deepening search will reach.

TIME_LIMIT: The maximum time (in seconds) the engine will spend thinking per move.

ASPIRATION_WINDOW: The window size for aspiration search, affecting how aggressively the engine searches for scores.

NULL_MOVE_R: Reduction factor for null move pruning.

LATE_MOVE_PRUNING_THRESHOLD: The threshold for applying late move pruning.

STATS: Set to True to print detailed search statistics during the engine's turn.

VALUES: Dictionary defining the material values for each piece type.

MOBILITY_BONUS: Bonuses for piece mobility.

ATTACK_WEIGHTS: Weights for king attack evaluation.

KING_ATTACK_SCALE: Scaling factor for king attack based on the number of attacking pieces.

PST: Piece-Square Tables for midgame ('mg') and endgame ('eg') evaluation.

##How it Works (Brief Overview)
NegaKnightV7 employs a Negamax algorithm with alpha-beta pruning as its core search function. This algorithm efficiently explores possible move sequences to find the best move for the current player.

Key enhancements include:

Iterative Deepening: The engine gradually increases its search depth, allowing it to return a reasonable move even if time runs out, and to leverage results from shallower searches.

Transposition Table: Stores the results of previously evaluated positions (identified by a Zobrist hash) to avoid redundant calculations and to enable faster lookups, especially for positions visited multiple times.

Quiescence Search: After the main Negamax search, a shallow quiescence search is performed to evaluate "noisy" positions (those with immediate captures or promotions) more accurately, preventing the horizon effect.

Move Ordering: Moves are sorted to try the most promising ones first. This significantly improves the effectiveness of alpha-beta pruning. It prioritizes hash moves, captures (using SEE), checks, and then moves from killer moves and history heuristic.

Positional Evaluation: The evaluate function calculates a score for a given board position based on various factors, providing a quantitative measure of who is "winning." This includes material balance, piece-square tables, pawn structure, king safety, and piece activity.
