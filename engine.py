import chess
import random
import time
import sys
from collections import defaultdict
import threading

MAX_DEPTH = 14
TIME_LIMIT = 5
ASPIRATION_WINDOW = 30
NULL_MOVE_R = 2
LATE_MOVE_PRUNING_THRESHOLD = 8

VALUES = {chess.PAWN: 100, chess.KNIGHT: 305, chess.BISHOP: 333, chess.ROOK: 563, chess.QUEEN: 950, chess.KING: 0}

MOBILITY_BONUS = [0, 4, 5, 2, 1]
ATTACK_WEIGHTS = [0] * 7
ATTACK_WEIGHTS[chess.KNIGHT] = 20
ATTACK_WEIGHTS[chess.BISHOP] = 40
ATTACK_WEIGHTS[chess.ROOK] = 80
ATTACK_WEIGHTS[chess.QUEEN] = 160
KING_ATTACK_SCALE = [0, 0, 50, 75, 88, 94, 97, 99, 99, 99, 99, 99, 99, 99, 99, 99]

import random as rnd
rnd.seed(42)

ZOBRIST_PIECES = {}
ZOBRIST_CASTLING = [rnd.getrandbits(64) for _ in range(16)]
ZOBRIST_EP = [rnd.getrandbits(64) for _ in range(8)]
ZOBRIST_TURN = rnd.getrandbits(64)

for piece_type in range(1, 7):
    for color in range(2):
        ZOBRIST_PIECES[(piece_type, color)] = [rnd.getrandbits(64) for _ in range(64)]

def compute_zobrist_hash(board):
    h = 0
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            h ^= ZOBRIST_PIECES[(piece.piece_type, piece.color)][square]

    castling_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE): castling_rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castling_rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK): castling_rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castling_rights |= 8
    h ^= ZOBRIST_CASTLING[castling_rights]

    if board.ep_square:
        h ^= ZOBRIST_EP[chess.square_file(board.ep_square)]

    if board.turn == chess.BLACK:
        h ^= ZOBRIST_TURN

    return h

PST = {
    chess.PAWN: {
        'mg': [0,0,0,0,0,0,0,0, 2,4,6,8,8,6,4,2, 2,4,8,12,12,8,4,2, 4,8,12,16,16,12,8,4, 6,10,15,20,20,15,10,6, 8,12,18,24,24,18,12,8, 50,50,50,50,50,50,50,50, 0,0,0,0,0,0,0,0],
        'eg': [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 30,30,30,30,30,30,30,30, 20,20,20,20,20,20,20,20, 10,10,10,10,10,10,10,10, 5,5,5,5,5,5,5,5, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]
    },
    chess.KNIGHT: {
        'mg': [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
        'eg': [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50]
    },
    chess.BISHOP: {
        'mg': [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
        'eg': [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20]
    },
    chess.ROOK: {
        'mg': [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
        'eg': [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0]
    },
    chess.QUEEN: {
        'mg': [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
        'eg': [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20]
    },
    chess.KING: {
        'mg': [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
        'eg': [-50,-40,-30,-20,-20,-30,-40,-50, -30,-20,-10,0,0,-10,-20,-30, -30,-10,20,30,30,20,-10,-30, -30,-10,30,40,40,30,-10,-30, -30,-10,30,40,40,30,-10,-30, -30,-10,20,30,30,20,-10,-30, -30,-30,0,0,0,0,-30,-30, -50,-30,-30,-30,-30,-30,-30,-50]
    }
}

class TranspositionTable:
    def __init__(self, size_mb=128):
        self.size = (size_mb * 1024 * 1024) // 32
        self.table = {}
        self.hits = 0
        self.age = 0

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.age += 1

    def get(self, key, depth, alpha, beta):
        if key in self.table:
            entry = self.table[key]
            if entry['depth'] >= depth:
                self.hits += 1
                val, flag = entry['value'], entry['flag']
                if flag == 'EXACT': return val
                if flag == 'LOWER' and val >= beta: return val
                if flag == 'UPPER' and val <= alpha: return val
        return None

    def store(self, key, depth, value, flag, move=None):
        if len(self.table) >= self.size:
            oldest_key = min(self.table.keys(), key=lambda k: self.table[k].get('age', 0))
            del self.table[oldest_key]

        self.table[key] = {'depth': depth, 'value': value, 'flag': flag, 'move': move, 'age': self.age}

    def get_move(self, key):
        return self.table.get(key, {}).get('move')

tt = TranspositionTable()

def phase_value(board):
    phase = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.piece_type == chess.KNIGHT: phase += 1
            elif piece.piece_type == chess.BISHOP: phase += 1
            elif piece.piece_type == chess.ROOK: phase += 2
            elif piece.piece_type == chess.QUEEN: phase += 4
    return min(phase, 24)

def is_endgame(board):
    return phase_value(board) < 8

def see(board, move):
    to_sq = move.to_square
    from_sq = move.from_square
    piece = board.piece_at(from_sq)

    if not piece:
        return 0

    captured = board.piece_at(to_sq)
    gain = [VALUES.get(captured.piece_type, 0) if captured else 0]

    if move.promotion:
        gain[0] += VALUES[move.promotion] - VALUES[chess.PAWN]

    attackers = board.attackers(chess.WHITE, to_sq) | board.attackers(chess.BLACK, to_sq)
    attackers = attackers - {from_sq}

    side = not piece.color
    attacking_piece_value = VALUES[piece.piece_type]

    d = 1
    while attackers and d < 32:
        gain.append(attacking_piece_value - gain[d-1])

        found = False
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            pieces = board.pieces(pt, side)
            for sq in pieces:
                if sq in attackers:
                    attacker_sq = sq
                    attackers.remove(attacker_sq)
                    attacking_piece_value = VALUES[pt]
                    found = True
                    break
            if found:
                break
        if not found:
            break

        side = not side
        d += 1

    while d > 1:
        d -= 1
        gain[d-1] = -max(-gain[d-1], gain[d])

    return gain[0]

def evaluate_king_attack(board, color):
    king_sq = board.king(not color)
    if not king_sq:
        return 0

    attack_count = 0
    attack_value = 0

    king_zone = chess.BB_KING_ATTACKS[king_sq]

    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, color):
            attacks = board.attacks(sq) & king_zone
            if attacks:
                attack_count += 1
                attack_value += ATTACK_WEIGHTS[piece_type] * bin(attacks).count('1')

    if attack_count < 2:
        return 0

    attack_index = min(attack_count, 15)
    return (attack_value * KING_ATTACK_SCALE[attack_index]) // 100

def evaluate_mobility(board, color):
    mobility = 0

    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, color):
            moves = len(board.attacks(sq))
            if piece_type == chess.KNIGHT:
                mobility += max(0, moves - 4) * MOBILITY_BONUS[1]
            elif piece_type == chess.BISHOP:
                mobility += max(0, moves - 7) * MOBILITY_BONUS[2]
            elif piece_type == chess.ROOK:
                mobility += max(0, moves - 7) * MOBILITY_BONUS[3]
            elif piece_type == chess.QUEEN:
                mobility += max(0, moves - 14) * MOBILITY_BONUS[4]

    return mobility

def evaluate_pawn_structure(board):
    score = 0
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    for sq in white_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        is_passed = True
        for enemy_sq in black_pawns:
            enemy_file = chess.square_file(enemy_sq)
            enemy_rank = chess.square_rank(enemy_sq)
            if abs(enemy_file - file) <= 1 and enemy_rank > rank:
                is_passed = False
                break

        if is_passed:
            passed_bonus = [0, 5, 10, 20, 35, 60, 100, 200][rank]
            score += passed_bonus

        if bin(white_pawns & chess.BB_FILES[file]).count('1') > 1:
            score -= 15

        isolated = True
        for adj_file in [file-1, file+1]:
            if 0 <= adj_file < 8 and white_pawns & chess.BB_FILES[adj_file]:
                isolated = False
                break
        if isolated:
            score -= 10

    for sq in black_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        is_passed = True
        for enemy_sq in white_pawns:
            enemy_file = chess.square_file(enemy_sq)
            enemy_rank = chess.square_rank(enemy_sq)
            if abs(enemy_file - file) <= 1 and enemy_rank < rank:
                is_passed = False
                break

        if is_passed:
            passed_bonus = [0, 5, 10, 20, 35, 60, 100, 200][7-rank]
            score -= passed_bonus

        if bin(black_pawns & chess.BB_FILES[file]).count('1') > 1:
            score += 15

        isolated = True
        for adj_file in [file-1, file+1]:
            if 0 <= adj_file < 8 and black_pawns & chess.BB_FILES[adj_file]:
                isolated = False
                break
        if isolated:
            score += 10

    return score

def evaluate_outposts(board):
    score = 0

    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if rank >= 4:
            protected = False
            for pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
                if abs(chess.square_file(pawn_sq) - file) == 1 and chess.square_rank(pawn_sq) == rank - 1:
                    protected = True
                    break
            if protected:
                score += 15

    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if rank <= 3:
            protected = False
            for pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
                if abs(chess.square_file(pawn_sq) - file) == 1 and chess.square_rank(pawn_sq) == rank + 1:
                    protected = True
                    break
            if protected:
                score -= 15

    return score

def evaluate_rook_files(board):
    score = 0

    for sq in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        file_mask = chess.BB_FILES[file]

        white_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) & file_mask)
        black_pawns = bool(board.pieces(chess.PAWN, chess.BLACK) & file_mask)

        if not white_pawns and not black_pawns:
            score += 25
        elif not white_pawns:
            score += 15

        if rank == 6:
            score += 20

    for sq in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        file_mask = chess.BB_FILES[file]

        white_pawns = bool(board.pieces(chess.PAWN, chess.WHITE) & file_mask)
        black_pawns = bool(board.pieces(chess.PAWN, chess.BLACK) & file_mask)

        if not white_pawns and not black_pawns:
            score -= 25
        elif not black_pawns:
            score -= 15

        if rank == 1:
            score -= 20

    return score

def evaluate_minor_piece_activity(board):
    score = 0
    minor_piece_moves = defaultdict(int)
    piece_move_counts = defaultdict(int)

    for i, move in enumerate(board.move_stack):
        if i >= 10:
            break
        piece = board.piece_type_at(move.from_square)
        if piece in [chess.KNIGHT, chess.BISHOP]:
            piece_move_counts[piece] += 1
            minor_piece_moves[move.from_square] += 1

    for piece_type in [chess.KNIGHT, chess.BISHOP]:
        for sq in board.pieces(piece_type, chess.WHITE):
            penalty = minor_piece_moves.get(sq, 0) * 5
            score -= penalty
        for sq in board.pieces(piece_type, chess.BLACK):
            penalty = minor_piece_moves.get(sq, 0) * 5
            score += penalty

    return score

def evaluate_king_safety(board):
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq:
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)

            pawn_shield = 0
            for file_offset in [-1, 0, 1]:
                shield_file = king_file + file_offset
                if 0 <= shield_file < 8:
                    if color == chess.WHITE:
                        shield_rank = king_rank - 1
                        if shield_rank >= 0:
                            sq = chess.square(shield_file, shield_rank)
                            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                                pawn_shield += 1
                        shield_rank = king_rank - 2
                        if shield_rank >= 0:
                            sq = chess.square(shield_file, shield_rank)
                            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                                pawn_shield += 1
                    else:
                        shield_rank = king_rank + 1
                        if shield_rank < 8:
                            sq = chess.square(shield_file, shield_rank)
                            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                                pawn_shield += 1
                        shield_rank = king_rank + 2
                        if shield_rank < 8:
                            sq = chess.square(shield_file, shield_rank)
                            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                                pawn_shield += 1

            if pawn_shield < 3:
                penalty = (3 - pawn_shield) * 25
                if color == chess.WHITE:
                    score -= penalty
                else:
                    score += penalty

    return score

def evaluate_center_control(board):
    score = 0
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]

    for square in center_squares:
        attackers_white = len(board.attackers(chess.WHITE, square))
        attackers_black = len(board.attackers(chess.BLACK, square))
        score += (attackers_white - attackers_black) * 5

    return score

def evaluate(board):
    if board.is_checkmate():
        return -30000 if board.turn else 30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    endgame = is_endgame(board)
    phase = phase_value(board)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            value = VALUES[piece.piece_type]

            mg_pst = PST[piece.piece_type]['mg'][sq if piece.color else chess.square_mirror(sq)]
            eg_pst = PST[piece.piece_type]['eg'][sq if piece.color else chess.square_mirror(sq)]
            pst_value = ((mg_pst * phase) + (eg_pst * (24 - phase))) // 24

            if piece.color:
                score += value + pst_value
            else:
                score -= value + pst_value

    score += evaluate_pawn_structure(board)
    score += evaluate_outposts(board)
    score += evaluate_rook_files(board)
    score += evaluate_minor_piece_activity(board)
    score += evaluate_king_safety(board)
    score += evaluate_center_control(board)

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30

    score += evaluate_mobility(board, chess.WHITE)
    score -= evaluate_mobility(board, chess.BLACK)

    if not endgame:
        score += evaluate_king_attack(board, chess.WHITE)
        score -= evaluate_king_attack(board, chess.BLACK)

    return score if board.turn else -score

killer_moves = [[] for _ in range(64)]
history_heuristic = defaultdict(int)

def order_moves(board, moves, hash_move=None, depth=0):
    def move_score(move):
        if hash_move and move == hash_move:
            return 30000

        if board.is_capture(move) or move.promotion:
            see_value = see(board, move)
            if see_value >= 0:
                return 20000 + see_value
            else:
                return 5000 + see_value

        if board.gives_check(move):
            return 15000

        if depth < 64 and move in killer_moves[depth]:
            return 10000

        return history_heuristic[move]

    return sorted(moves, key=move_score, reverse=True)

def quiesce(board, alpha, beta, start_time, stats):
    stats['nodes'] += 1
    stats['qnodes'] += 1

    if time.time() - start_time > TIME_LIMIT:
        return alpha

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = []
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion:
            see_value = see(board, move)
            if see_value >= -50:
                captures.append((move, see_value))

    captures.sort(key=lambda x: x[1], reverse=True)

    for move, _ in captures:
        board.push(move)
        score = -quiesce(board, -beta, -alpha, start_time, stats)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def negamax(board, depth, alpha, beta, start_time, ply, stats):
    stats['nodes'] += 1

    if time.time() - start_time > TIME_LIMIT:
        return alpha

    if board.is_checkmate():
        return -30000 + ply
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    board_hash = compute_zobrist_hash(board)
    tt_value = tt.get(board_hash, depth, alpha, beta)
    if tt_value is not None:
        stats['tt_hits'] += 1
        return tt_value

    if depth >= 3 and not board.is_check() and any(board.pieces(pt, board.turn) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
        reduction = NULL_MOVE_R + min(3, depth // 6)
        board.push(chess.Move.null())
        null_score = -negamax(board, depth - 1 - reduction, -beta, -beta + 1, start_time, ply + 1, stats)
        board.pop()

        if null_score >= beta:
            return null_score

    if depth == 0:
        return quiesce(board, alpha, beta, start_time, stats)

    alpha_orig = alpha
    hash_move = tt.get_move(board_hash)
    moves = list(board.legal_moves)

    if not moves:
        return -30000 + ply if board.is_check() else 0

    moves = order_moves(board, moves, hash_move, ply)

    best_value = -31000
    best_move = None

    for i, move in enumerate(moves):
        if depth >= 2 and i >= LATE_MOVE_PRUNING_THRESHOLD and not board.is_check() and not board.is_capture(move) and not move.promotion and not board.gives_check(move):
            continue

        board.push(move)

        reduction = 0
        if depth >= 3 and i >= 4 and not board.is_check() and not board.is_capture(move) and not move.promotion:
            reduction = min(depth // 3, 2)

        if i == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, start_time, ply + 1, stats)
        else:
            score = -negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, start_time, ply + 1, stats)
            if score > alpha and reduction > 0:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, start_time, ply + 1, stats)
            if score > alpha and score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, start_time, ply + 1, stats)

        board.pop()

        if score > best_value:
            best_value = score
            best_move = move

        if score > alpha:
            alpha = score
            if not board.is_capture(move) and not move.promotion:
                if ply < 64:
                    if move not in killer_moves[ply]:
                        killer_moves[ply].insert(0, move)
                        if len(killer_moves[ply]) > 2:
                            killer_moves[ply].pop()
                history_heuristic[move] += depth * depth

        if alpha >= beta:
            break

    if best_value <= alpha_orig:
        flag = 'UPPER'
    elif best_value >= beta:
        flag = 'LOWER'
    else:
        flag = 'EXACT'

    tt.store(board_hash, depth, best_value, flag, best_move)
    return best_value

def iterative_deepening(board, max_depth, time_limit):
    start_time = time.time()
    best_move = None
    best_score = 0

    for depth in range(max_depth):
        killer_moves[depth].clear()

    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit * 0.85:
            break

        stats = {'nodes': 0, 'qnodes': 0, 'tt_hits': 0}
        start_depth_time = time.time()

        if depth <= 3:
            alpha, beta = -31000, 31000
        else:
            alpha, beta = best_score - ASPIRATION_WINDOW, best_score + ASPIRATION_WINDOW

        for attempt in range(4):
            score = negamax(board, depth, alpha, beta, start_time, 0, stats)

            if score <= alpha:
                alpha = best_score - ASPIRATION_WINDOW * (2 ** attempt)
                if alpha < -31000:
                    alpha = -31000
            elif score >= beta:
                beta = best_score + ASPIRATION_WINDOW * (2 ** attempt)
                if beta > 31000:
                    beta = 31000
            else:
                best_score = score
                move = tt.get_move(compute_zobrist_hash(board))
                if move and move in board.legal_moves:
                    best_move = move
                break

        if abs(best_score) > 29000:
            break

        pv = []
        temp_board = board.copy()
        for i in range(depth):
            h = compute_zobrist_hash(temp_board)
            move = tt.get_move(h)
            if move is None or move not in temp_board.legal_moves:
                break
            pv.append(move)
            temp_board.push(move)

        depth_time = time.time() - start_depth_time
        nps = int(stats['nodes'] / depth_time) if depth_time > 0 else 0

        # Convert score to white's perspective for UCI
        white_score = best_score if board.turn == chess.WHITE else -best_score
        print(f"info depth {depth} score cp {int(white_score)} time {int(depth_time*1000)} nodes {stats['nodes']} nps {nps} pv {' '.join(m.uci() for m in pv)}")

    return best_move or random.choice(list(board.legal_moves))

class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.search_stop = threading.Event()
        self.move_overhead = 50  # milliseconds

    def uci_loop(self):
        print("id name NegaKnightV7")
        print("id author Anish")
        print("uciok", flush=True)

        while True:
            line = sys.stdin.readline().strip()
            if line == "quit":
                break
            elif line == "uci":
                print("id name NegaKnightV7")
                print("id author Anish")
                print("uciok", flush=True)
            elif line == "isready":
                print("readyok", flush=True)
            elif line == "ucinewgame":
                tt.clear()
                global killer_moves, history_heuristic
                killer_moves = [[] for _ in range(64)]
                history_heuristic = defaultdict(int)
            elif line.startswith("position"):
                self.handle_position(line)
            elif line.startswith("go"):
                self.handle_go(line)
            elif line == "stop":
                self.search_stop.set()
            elif line == "print":
                print(self.board)

    def handle_position(self, line):
        parts = line.split()
        if parts[1] == 'startpos':
            self.board = chess.Board()
            moves_idx = 2
        elif parts[1] == 'fen':
            fen = ' '.join(parts[2:8])
            self.board = chess.Board(fen)
            moves_idx = 8
        else:
            return

        if moves_idx < len(parts) and parts[moves_idx] == 'moves':
            for move_str in parts[moves_idx+1:]:
                move = chess.Move.from_uci(move_str)
                self.board.push(move)

    def handle_go(self, line):
        self.search_stop.clear()
        global TIME_LIMIT, MAX_DEPTH

        parts = line.split()
        max_depth = MAX_DEPTH
        movetime = None
        wtime = btime = winc = binc = None

        i = 1
        while i < len(parts):
            if parts[i] == 'depth':
                max_depth = int(parts[i+1])
                i += 1
            elif parts[i] == 'movetime':
                movetime = int(parts[i+1])
                i += 1
            elif parts[i] == 'wtime':
                wtime = int(parts[i+1])
                i += 1
            elif parts[i] == 'btime':
                btime = int(parts[i+1])
                i += 1
            elif parts[i] == 'winc':
                winc = int(parts[i+1])
                i += 1
            elif parts[i] == 'binc':
                binc = int(parts[i+1])
                i += 1
            i += 1

        time_limit = 5.0  # default to 5 seconds

        if movetime is not None:
            time_limit = movetime / 1000.0
        elif wtime is not None and btime is not None:
            time_left = wtime if self.board.turn == chess.WHITE else btime
            increment = winc if self.board.turn == chess.WHITE else binc
            time_limit = max(0.1, min(time_left / 1000.0 / 40.0, 5.0))
            time_limit -= self.move_overhead / 1000.0
            time_limit = max(0.1, time_limit)

        TIME_LIMIT = time_limit
        MAX_DEPTH = max_depth

        # Start search in a separate thread
        search_thread = threading.Thread(target=self.start_search, args=(time_limit, max_depth))
        search_thread.daemon = True
        search_thread.start()

    def start_search(self, time_limit, max_depth):
        best_move = iterative_deepening(self.board, max_depth, time_limit)
        print(f"bestmove {best_move.uci()}", flush=True)

if __name__ == "__main__":
    engine = UCIEngine()
    engine.uci_loop()