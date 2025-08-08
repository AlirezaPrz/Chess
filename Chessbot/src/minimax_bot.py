import random
import chess_engine
import math
import numpy as np

# ── static data (compute once) ───────────────────────────────────────────
# Material values
_MATERIAL = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0}

# Piece-square tables (white perspective, black uses row-flip)
_PST = {
    'P': np.array([
        [0,   0,   0,   0,   0,   0,   0,  0],
        [0,   0,   0,   0,   0,  0,   0,   0],
        [5,   5,  10,   0,   0, 10,   5,   5],
        [0,  15,  15,  25,  25, 15,  15,   0],
        [0,   0,  30,  30,  30, 30,   0,   0],
        [0,   0,   0,  40,  40,  0,   0,   0],
        [0,   0,   0,   0,   0,  0,   0,   0],
        [0,   0,   0,   0,   0,  0,   0,   0]
    ]),
    'N': np.array([
        [-40,-15,-10,-10,-10,-10,-15,-40],
        [-15,-10,  0,  0,  0,  0,-10,-15],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10,  0, 10, 20, 20, 10,  0,-10],
        [-10,  0, 10, 20, 20, 10,  0,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-15,-10,  0,  0,  0,  0,-10,-15],
        [-40,-15,-10,-10,-10,-10,-15,-40]
    ]),
    'B': np.zeros((8, 8), dtype=int),
    'R': np.zeros((8, 8), dtype=int),
    'Q': np.zeros((8, 8), dtype=int),
    'K': np.zeros((8, 8), dtype=int)  # filled later (mid/late game split)
}
# End-game king PST (king becomes strong in centre)
_PST['K_END'] = np.array([
    [-50,-30,-30,-30,-30,-30,-30,-50],
    [-30,-10,-10,-10,-10,-10,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-20,-10,-10,-10,-10,-20,-30],
    [-50,-40,-30,-20,-20,-30,-40,-50]
])

def _flip(arr):
    """Flip an array vertically (for black's perspective PST)."""
    return np.flipud(arr)

# Cache flipped tables for black
_PST_B = {pt: _flip(tab) for pt, tab in _PST.items()}
_PST_B['K_END'] = _flip(_PST['K_END'])

# ── static evaluator ─────────────────────────────────────────────────────
def evaluate_board(gs: chess_engine.GameState) -> float:
    """
    Evaluate the chess position and return a score in pawns.
    Positive score favors White, negative favors Black.
    Considers material, piece-square tables, pawn structure, piece activity, and king safety.
    """
    board = gs.board
    score = 0
    white_bishops = 0
    black_bishops = 0
    white_rooks = []
    black_rooks = []
    pawn_files_w = [0]*8
    pawn_files_b = [0]*8
    white_king = None
    black_king = None
    white_queen = None
    black_queen = None
    white_knights = []
    black_knights = []
    white_pawn_positions = []
    black_pawn_positions = []
    white_bishops_positions = []
    black_bishops_positions = []
    total_material = 0

    # --- Scan board and accumulate values ---
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == chess_engine.EMPTY_SQUARE:
                continue
            color = 1 if piece[0] == 'w' else -1
            ptype = piece[1]

            # Material value
            val = _MATERIAL[ptype]
            score += color * val
            total_material += val

            # Piece-square table value
            if color == 1:
                pst_val = _PST[ptype][r, c]
            else:
                pst_val = _PST_B[ptype][r, c]
            score += color * pst_val

            # Track pieces for special heuristics
            if ptype == 'B':
                if color == 1:
                    white_bishops += 1
                    white_bishops_positions.append((r, c))
                else:
                    black_bishops += 1
                    black_bishops_positions.append((r, c))
            elif ptype == 'R':
                if color == 1:
                    white_rooks.append(c)
                else:
                    black_rooks.append(c)
            elif ptype == 'N':
                if color == 1:
                    white_knights.append((r, c))
                else:
                    black_knights.append((r, c))
            elif ptype == 'Q':
                if color == 1:
                    white_queen = (r, c)
                else:
                    black_queen = (r, c)
            elif ptype == 'P':
                if color == 1:
                    pawn_files_w[c] += 1
                    white_pawn_positions.append((r, c))
                else:
                    pawn_files_b[c] += 1
                    black_pawn_positions.append((r, c))
            elif ptype == 'K':
                if color == 1:
                    white_king = (r, c)
                else:
                    black_king = (r, c)

    # Bishop pair bonus
    if white_bishops >= 2:
        score += 35
    if black_bishops >= 2:
        score -= 35

    # Rook on open or semi-open file
    for fc in white_rooks:
        if pawn_files_w[fc] == 0:
            score += 20 if pawn_files_b[fc] else 50
    for fc in black_rooks:
        if pawn_files_b[fc] == 0:
            score -= 20 if pawn_files_w[fc] else 50

    # Determine game phase for king table
    end_game = total_material <= 2600

    # King positional and safety
    def _king_bonus(pos, side_color):
        r, c = pos
        table = 'K_END' if end_game else 'K'
        return (_PST if side_color == 1 else _PST_B)[table][r, c]

    if white_king:
        score += _king_bonus(white_king, 1)
        wr, wc = white_king
        if wr == 7:
            for dc in (-1, 0, 1):
                f = wc + dc
                if 0 <= f < 8 and board[6][f] != 'wP':
                    score -= 10

    if black_king:
        score -= _king_bonus(black_king, -1)
        br, bc = black_king
        if br == 0:
            for dc in (-1, 0, 1):
                f = bc + dc
                if 0 <= f < 8 and board[1][f] != 'bP':
                    score += 10
                    
    if gs.white_castled:
        score -= 50
    if gs.black_castled:
        score += 50

    # Additional positional evaluations
    # Pawn structure: doubled, isolated, passed pawns
    # Doubled pawns
    for f in range(8):
        if pawn_files_w[f] > 1:
            score -= (pawn_files_w[f] - 1) * 5
        if pawn_files_b[f] > 1:
            score += (pawn_files_b[f] - 1) * 5
    # Isolated pawns
    for f in range(8):
        if pawn_files_w[f] > 0 and (f == 0 or pawn_files_w[f-1] == 0) and (f == 7 or pawn_files_w[f+1] == 0):
            score -= pawn_files_w[f] * 5
        if pawn_files_b[f] > 0 and (f == 0 or pawn_files_b[f-1] == 0) and (f == 7 or pawn_files_b[f+1] == 0):
            score += pawn_files_b[f] * 5
    # Passed pawns
    for (pr, pc) in white_pawn_positions:
        passed = True
        for (bpr, bpc) in black_pawn_positions:
            if bpc in (pc-1, pc, pc+1) and bpr < pr:
                passed = False
                break
        if passed:
            white_rank = 8 - pr
            bonus = 0
            if white_rank > 2:
                bonus = (white_rank - 2) * 15
            if end_game:
                bonus *= 2
            score += bonus
    for (bpr, bpc) in black_pawn_positions:
        passed = True
        for (pr, pc) in white_pawn_positions:
            if pc in (bpc-1, bpc, bpc+1) and pr > bpr:
                passed = False
                break
        if passed:
            black_rank = bpr + 1
            bonus = 0
            if black_rank > 2:
                bonus = (black_rank - 2) * 15
            if end_game:
                bonus *= 2
            score -= bonus
    
    # Knight outposts
    for (nr, nc) in white_knights:
        if nr <= 3:
            safe = True
            for (bpr, bpc) in black_pawn_positions:
                if bpc in (nc-1, nc+1) and bpr < nr:
                    safe = False
                    break
            if not safe:
                continue
            supported = ((nr+1, nc-1) in white_pawn_positions or (nr+1, nc+1) in white_pawn_positions)
            score += 30 if supported else 15
    for (nr, nc) in black_knights:
        if nr >= 4:
            safe = True
            for (pr, pc) in white_pawn_positions:
                if pc in (nc-1, nc+1) and pr > nr:
                    safe = False
                    break
            if not safe:
                continue
            supported = ((nr-1, nc-1) in black_pawn_positions or (nr-1, nc+1) in black_pawn_positions)
            score -= 30 if supported else 15
    
    # Rook on 7th rank
    for c in range(8):
        if board[1][c] == 'wR':
            score += 20
        if board[6][c] == 'bR':
            score -= 20
    # Early queen development penalty
    if gs.turn < 12:
        undeveloped_white_minors = 0
        if board[7][1] == 'wN':
            undeveloped_white_minors += 1
        if board[7][6] == 'wN':
            undeveloped_white_minors += 1
        if board[7][2] == 'wB':
            undeveloped_white_minors += 1
        if board[7][5] == 'wB':
            undeveloped_white_minors += 1
        if white_queen is not None and white_queen != (7, 3) and undeveloped_white_minors > 0:
            score -= 30
        undeveloped_black_minors = 0
        if board[0][1] == 'bN':
            undeveloped_black_minors += 1
        if board[0][6] == 'bN':
            undeveloped_black_minors += 1
        if board[0][2] == 'bB':
            undeveloped_black_minors += 1
        if board[0][5] == 'bB':
            undeveloped_black_minors += 1
        if black_queen is not None and black_queen != (0, 3) and undeveloped_black_minors > 0:
            score += 30
    # Bishop blocked (immobile) penalty
    for (br, bc) in white_bishops_positions:
        if br > 0:
            block_count = 0
            if bc > 0 and board[br-1][bc-1] != chess_engine.EMPTY_SQUARE and board[br-1][bc-1][0] == 'w':
                block_count += 1
            if bc < 7 and board[br-1][bc+1] != chess_engine.EMPTY_SQUARE and board[br-1][bc+1][0] == 'w':
                block_count += 1
            if block_count == 2:
                score -= 25
    for (br, bc) in black_bishops_positions:
        if br < 7:
            block_count = 0
            if bc > 0 and board[br+1][bc-1] != chess_engine.EMPTY_SQUARE and board[br+1][bc-1][0] == 'b':
                block_count += 1
            if bc < 7 and board[br+1][bc+1] != chess_engine.EMPTY_SQUARE and board[br+1][bc+1][0] == 'b':
                block_count += 1
            if block_count == 2:
                score += 25

    return score / 100.0

def minimax(gs: chess_engine.GameState, depth: int, alpha: float, beta: float) -> float:
    """
    Minimax search with alpha-beta pruning. Returns the evaluation score from White's perspective.
    """
    if depth == 0:
        return evaluate_board(gs)
    moves = gs.get_valid_moves()
    if len(moves) == 0:
        # Terminal position: checkmate or stalemate
        if gs.white_to_move:
            return -10000.0 - depth if gs.in_check('w') else 0.0
        else:
            return 10000.0 + depth if gs.in_check('b') else 0.0
    if gs.white_to_move:
        max_eval = -math.inf
        for move in moves:
            gs.make_move(move, update=False)
            eval_score = minimax(gs, depth - 1, alpha, beta)
            gs.undo_move()
            if eval_score > max_eval:
                max_eval = eval_score
            if eval_score > alpha:
                alpha = eval_score
            if beta <= alpha:
                break  # alpha-beta pruning
        return max_eval
    else:
        min_eval = math.inf
        for move in moves:
            gs.make_move(move, update=False)
            eval_score = minimax(gs, depth - 1, alpha, beta)
            gs.undo_move()
            if eval_score < min_eval:
                min_eval = eval_score
            if eval_score < beta:
                beta = eval_score
            if beta <= alpha:
                break
        return min_eval

def find_best_move(gs: chess_engine.GameState, depth: int) -> chess_engine.Move:
    """
    Search the given number of plies and return the best move for the side to move.
    """
    best_move = None
    if gs.white_to_move:
        best_value = -math.inf
        for move in gs.get_valid_moves():
            gs.make_move(move, update=False)
            move_val = minimax(gs, depth - 1, -math.inf, math.inf)
            gs.undo_move()
            if move_val > best_value:
                best_value = move_val
                best_move = move
    else:
        best_value = math.inf
        for move in gs.get_valid_moves():
            gs.make_move(move, update=False)
            move_val = minimax(gs, depth - 1, -math.inf, math.inf)
            gs.undo_move()
            if move_val < best_value:
                best_value = move_val
                best_move = move
    return best_move
