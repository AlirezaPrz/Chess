import numpy as np

# ---------------------------------------------------------------------#
# Constants and Initial Setup
# ---------------------------------------------------------------------#
EMPTY_SQUARE = "--"  # Two dashes denote an empty board cell
BOARD_SIZE = 8       # Board dimension (8x8)

class GameState:
    """
    A container for all dynamic information that defines a chess position.
    It maintains the board state, turn information, castling rights, en passant rights,
    half-move clock for the 50-move rule, and flags for checkmate/stalemate/draw.
    It can generate moves and apply or undo them, updating the state accordingly.
    """

    def __init__(self) -> None:
        # Initialize the board with the standard chess starting position.
        # Board is a 2D numpy array of strings; each piece is represented by 
        # a two-character code: ('w' or 'b' for color, followed by 'P', 'R', 'N', 'B', 'Q', or 'K').
        self.board = np.array([
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP"] * 8,
            [EMPTY_SQUARE] * 8,
            [EMPTY_SQUARE] * 8,
            [EMPTY_SQUARE] * 8,
            [EMPTY_SQUARE] * 8,
            ["wP"] * 8,
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ])

        # Side to move: True for white, False for black.
        self.white_to_move = True
        self.move_log: list[Move] = []  # Log of moves played (for history/undo)
        # Game termination flags
        self.checkmate = False
        self.stalemate = False
        self.draw = False
        self.threefold = False
        self.insuf_mat = False
        self.fifty_rule = False
        self.winner = None  # "w" or "b" if one side won, None if draw or ongoing

        # Castling rights: track if each side can still castle king-side (ks) or queen-side (qs).
        self.current_castling_rights = {"wks": True, "wqs": True, "bks": True, "bqs": True}
        self.castle_rights_log = [self.current_castling_rights.copy()]  # keep history of castling rights for undo

        # King positions (for quick check calculations)
        self.white_king_location = (7, 4)
        self.white_king_location_log = [(7, 4)]
        self.black_king_location = (0, 4)
        self.black_king_location_log = [(0, 4)]

        # En passant target: stores the square of a pawn that moved two steps in the last move (as a Move object).
        # This is used to validate en passant captures.
        self.en_passant_target: Move | None = None
        self.en_passant_target_log = [self.en_passant_target]  # history of en-passant targets for undo

        # 50-move rule counters
        self.turn = 0               # count of half-moves made in the game (incremented after each move)
        self.last_capture_turn = -1 # turn count when the last capture occurred
        self.last_pawn_turn = -1    # turn count when the last pawn move occurred
        self.last_capture_turn_log: list[int] = []
        self.last_pawn_turn_log: list[int] = []
        self.num_pieces = 32
        self.white_castled = False
        self.black_castled = False

        # Attributes used during move generation for efficiency
        self.inCheck = False
        self.pins: list[tuple[int, int, int, int]] = []   # list of pinned pieces and pin direction vectors
        self.checks: list[tuple[int, int, int, int]] = [] # list of checking pieces and check direction vectors
        
        # Add repetition tracking
        self.positions_count: dict[tuple, int] = {}
        # Record initial position occurrence
        self.positions_count[self._position_key()] = 1
        

    # ------------------------------------------------------------------#
    # Core Move Application and Reversal
    # ------------------------------------------------------------------#
    def make_move(self, move: "Move", update = True) -> None:
        """
        Apply one move (half-move) to the position and update board and state accordingly.
        `move` should be a Move object that is legal in the current position.
        If update_state is True, the game status (checkmate/stalemate/draw flags) will be updated after the move.
        """
        # Increment the half-move counter (turn index)
        self.turn += 1

        # 50-move rule bookkeeping: update last capture/pawn move turn indices
        if move.piece_captured != EMPTY_SQUARE:
            self.last_capture_turn = self.turn
            self.last_capture_turn_log.append(self.last_capture_turn)
            self.num_pieces-=1
            
        if move.piece_moved[1] == "P":  # if a pawn moved
            self.last_pawn_turn = self.turn
            self.last_pawn_turn_log.append(self.last_pawn_turn)

        # Handle special moves: en passant, promotion, castling
        # En passant capture: remove the pawn that was captured en passant
        if move.en_passant:
            # The pawn to be removed is not on the end square (which is empty in en passant)
            # but directly above/below the end square (same column as end, starting row of pawn).
            self.board[move.start_row][move.end_col] = EMPTY_SQUARE
            # Clear en_passant target (it is only valid immediately on the next move)
            self.en_passant_target = None
            self.num_pieces-=1
            
            self.last_capture_turn = self.turn
            self.last_capture_turn_log.append(self.last_capture_turn)

        # Promotion: we will promote to a Queen (this implementation auto-queens for simplicity)
        if move.promotion:
            self.board[move.end_row][move.end_col] = move.piece_moved[0] + "Q"
            self.board[move.start_row][move.start_col] = EMPTY_SQUARE
            # Finish move without further board updates (promotion done)
            self._update_castling_flags(move)
            self._finish_make(move, update = update)
            return

        # Castling move: move the rook to the other side of the king
        if move.castling:
            if move.end_col == 2:  # Queen-side castling
                self._castle(move, rook_from=0, rook_to=3)
            else:  # King-side castling
                self._castle(move, rook_from=7, rook_to=5)
            if move.piece_moved[0] == "w":
                self.white_castled = True
            else:
                self.black_castled = True

        # Normal move: move the piece from start square to end square
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.board[move.start_row][move.start_col] = EMPTY_SQUARE

        # Update castling rights flags if a king or rook moved (or rook was captured)
        self._update_castling_flags(move)

        # Update en passant target square for potential capture:
        if move.piece_moved[1] == "P" and abs(move.start_row - move.end_row) == 2:
            # A pawn moved two squares; mark the square it passed over as en-passant target
            self.en_passant_target = move
        else:
            self.en_passant_target = None
        # Record en-passant target in history for undo
        self.en_passant_target_log.append(self.en_passant_target)

        # Finalize the move: switch turn, log the move, and update game status if needed
        self._finish_make(move, update = update)

    def _finish_make(self, move: "Move", update) -> None:
        """Helper to finalize make_move once the board is updated: switch player turn, log move, and update game status."""
        # Switch whose turn it is
        self.white_to_move = not self.white_to_move
        # Log the move in the move history
        self.move_log.append(move)

        
        # After making the move, determine if the game has ended.
        # First, assume no special outcome:
        self.checkmate = False
        self.stalemate = False
        self.draw = False
        
        # Update repetition count for the new position
        pos_key = self._position_key()
        self.positions_count[pos_key] = self.positions_count.get(pos_key, 0) + 1

        if update:
            # Determine legal moves for the player who is now to move.
            moves = self.get_valid_moves()
            inCheck = self.in_check("w" if self.white_to_move else "b")
            if len(moves) == 0:
                # No legal moves available for the side to move.
                # Check whether that side's king is in check to decide checkmate vs stalemate.
                if inCheck:
                    # The side to move is in check with no moves: checkmate.
                    self.checkmate = True
                    # Record winner: if white_to_move (and in checkmate), that means white has no moves and is in check -> black wins.
                    self.winner = "b" if self.white_to_move else "w"
                else:
                    # No moves and not in check: stalemate (draw).
                    self.stalemate = True
            # Check other draw conditions (50-move rule, insufficient material, etc.)
            self.draw_game()
        

    def _castle(self, move: "Move", rook_from: int, rook_to: int) -> None:
        """
        Helper for castling moves: move the rook during castling.
        For king-side castling, rook_from is 7 and rook_to is 5 (rook moves from h-file to f-file).
        For queen-side castling, rook_from is 0 and rook_to is 3 (rook moves from a-file to d-file).
        """
        self.board[move.end_row][rook_to] = move.piece_moved[0] + "R"    # Place rook next to king
        self.board[move.end_row][rook_from] = EMPTY_SQUARE               # Clear the rook's original square

    def _update_castling_flags(self, move: "Move") -> None:
        """
        Update castling rights based on the move that was made.
        This will revoke castling rights if a king or rook moves, or if a rook is captured.
        """
        colour = move.piece_moved[0]  # 'w' or 'b'
        # If a king moved, lose both castling rights for that side.
        if move.piece_moved[1] == "K":
            if colour == "w":
                self.white_king_location = (move.end_row, move.end_col)  # update white king's position
                self.current_castling_rights["wks"] = False
                self.current_castling_rights["wqs"] = False
            else:
                self.black_king_location = (move.end_row, move.end_col)
                self.current_castling_rights["bks"] = False
                self.current_castling_rights["bqs"] = False
        
        self.white_king_location_log.append(self.white_king_location)
        self.black_king_location_log.append(self.black_king_location)

        # If a rook moved, revoke the corresponding castling right for that side.
        if move.piece_moved[1] == "R":
            if colour == "w":
                # White rook moved from either a1 (queen-side) or h1 (king-side)
                if move.start_row == 7 and move.start_col == 0:
                    self.current_castling_rights["wqs"] = False
                elif move.start_row == 7 and move.start_col == 7:
                    self.current_castling_rights["wks"] = False
            else:
                # Black rook moved from a8 or h8
                if move.start_row == 0 and move.start_col == 0:
                    self.current_castling_rights["bqs"] = False
                elif move.start_row == 0 and move.start_col == 7:
                    self.current_castling_rights["bks"] = False

        # If a rook was captured, revoke that castling right as well.
        if move.piece_captured in ("wR", "bR"):
            if move.piece_captured == "wR":
                # Determine which white rook was captured by its position
                if move.end_row == 7 and move.end_col == 0:
                    self.current_castling_rights["wqs"] = False
                elif move.end_row == 7 and move.end_col == 7:
                    self.current_castling_rights["wks"] = False
            else:  # captured piece is "bR"
                if move.end_row == 0 and move.end_col == 0:
                    self.current_castling_rights["bqs"] = False
                elif move.end_row == 0 and move.end_col == 7:
                    self.current_castling_rights["bks"] = False

        # Save the updated castling rights to the log (for undo functionality)
        self.castle_rights_log.append(self.current_castling_rights.copy())

    def undo_move(self) -> None:
        """
        Revert the last move played, restoring the board and all state to how it was before that move.
        This includes restoring castling rights, en passant status, half-move clocks, etc.
        """
        if not self.move_log:
            return  # No move to undo
        
        # Decrement repetition count of current position (about to undo)
        current_key = self._position_key()
        if current_key in self.positions_count:
            self.positions_count[current_key] -= 1
            if self.positions_count[current_key] == 0:
                del self.positions_count[current_key]

        move = self.move_log.pop()    # Get the last move from the log
        self.turn -= 1                # Decrement half-move count
        # Switch turn back to the side who played the move
        self.white_to_move = not self.white_to_move

        # Reset end-of-game flags
        self.checkmate = False
        self.stalemate = False
        self.draw = False
        self.winner = None

        # Revert the board to before the move:
        self.board[move.start_row][move.start_col] = move.piece_moved
        self.board[move.end_row][move.end_col] = move.piece_captured

        # If it was an en passant capture, we need to restore the pawn that was captured.
        if move.en_passant:
            # The pawn was captured from the square directly adjacent to the end square.
            # Place the pawn back on the square it was captured from.
            self.board[move.start_row][move.end_col] = "bP" if move.piece_moved[0] == "w" else "wP"
            self.num_pieces+=1
            # (The end square had been set to EMPTY_SQUARE in make_move, and we restored that above)
            
        if move.piece_captured != EMPTY_SQUARE:
            self.num_pieces+=1

        # If it was a promotion, revert the pawn (replace the promoted piece with the original pawn).
        if move.promotion:
            # move.piece_moved is the original pawn (e.g., 'wP'), piece_captured is what was on the promotion square.
            self.board[move.start_row][move.start_col] = move.piece_moved
            self.board[move.end_row][move.end_col] = move.piece_captured

        # If it was a castling move, move the rook back to its original square.
        if move.castling:
            if move.end_col == 2:
                # Queen-side castling: rook moved from 'rook_from=0' to 'rook_to=3'
                self.board[move.end_row][0] = move.piece_moved[0] + "R"
                self.board[move.end_row][3] = EMPTY_SQUARE
            else:
                # King-side castling: rook moved from 7 to 5
                self.board[move.end_row][7] = move.piece_moved[0] + "R"
                self.board[move.end_row][5] = EMPTY_SQUARE

        # Restore castling rights to the state before the undone move.
        if self.castle_rights_log:
            self.castle_rights_log.pop()  # remove the last entry (current state)
            # Reset current_castling_rights to the previous state (if exists)
            if self.castle_rights_log:
                self.current_castling_rights = self.castle_rights_log[-1].copy()

        # Restore en passant availability to previous state
        if self.en_passant_target_log:
            self.en_passant_target_log.pop()
            self.en_passant_target = self.en_passant_target_log[-1] if self.en_passant_target_log else None

        # Restore king locations by scanning the board for kings (efficient enough for an undo operation)
        self.white_king_location_log.pop()
        self.white_king_location = self.white_king_location_log[-1]
        self.black_king_location_log.pop()
        self.black_king_location = self.black_king_location_log[-1]

        # Recompute 50-move counters (half-move clocks) by scanning move log
        if self.last_capture_turn_log:
            self.last_capture_turn_log.pop()
            self.last_capture_turn = self.last_capture_turn_log[-1] if self.last_capture_turn_log else -1
        if self.last_pawn_turn_log:
            self.last_pawn_turn_log.pop()
            self.last_pawn_turn = self.last_pawn_turn_log[-1] if self.last_pawn_turn_log else -1


    def _position_key(self) -> tuple:
        """Generate a key representing the current position (board, turn, castling, en-passant)."""
        board_state = tuple(tuple(cell for cell in row) for row in self.board.tolist())
        # Include castling rights and en-passant target in key
        ep_square = None
        if self.en_passant_target:
            ep_square = ((self.en_passant_target.start_row + self.en_passant_target.end_row)//2,
                         self.en_passant_target.end_col)
        key = (board_state, self.white_to_move,
               self.current_castling_rights["wks"],
               self.current_castling_rights["wqs"],
               self.current_castling_rights["bks"],
               self.current_castling_rights["bqs"],
               ep_square)
        return key

    # ------------------------------------------------------------------#
    # Move Generation (Legal Moves)
    # ------------------------------------------------------------------#
    def get_valid_moves(self) -> list["Move"]:
        """
        Generate all legal moves for the current player to move.
        This function uses an optimized approach:
         - It first detects checks and pins on the king.
         - Generates pseudo-legal moves (moves that adhere to piece movement rules and basic board limits).
         - Filters out moves that leave the king in check.
        The result is a list of fully legal moves for the side whose turn it is.
        """
        # Preserve current castling rights (we may modify them during generation, so save & restore after)
        temp_castle_rights = self.current_castling_rights.copy()
        moves: list[Move] = []

        # Determine if the current player's king is in check and identify pinned pieces
        self.inCheck, self.pins, self.checks = self._check_for_pins_and_checks()

        king_row, king_col = (self.white_king_location if self.white_to_move else self.black_king_location)

        if self.inCheck:
            if len(self.checks) == 1:
                # Single check: generate all moves, then filter to only those that block or capture the checking piece, or move the king
                moves = self._get_all_possible_moves()  # pseudo-legal moves (takes pins into account)
                # Compute squares that can block the check or capture the attacker
                check_piece = self.checks[0]  # (row, col, dir_row, dir_col) of the single checking piece
                check_row, check_col, check_dir_r, check_dir_c = check_piece
                # Squares between king and checking piece (inclusive of the checking piece's square)
                valid_squares = [(check_row, check_col)]
                # If the attacker is a sliding piece (rook, bishop, or queen), it can be blocked.
                piece_type = self.board[check_row][check_col][1]
                if piece_type not in ("N",):  # knights cannot be blocked
                    # Add all squares on the line between king and attacker to valid_squares
                    for i in range(1, BOARD_SIZE):
                        sq = (king_row + check_dir_r * i, king_col + check_dir_c * i)
                        valid_squares.append(sq)
                        if sq[0] == check_row and sq[1] == check_col:
                            break  # reached the attacker's position
                # Filter out any move that doesn't either move the king or move to one of the valid blocking/capturing squares
                moves = [mv for mv in moves if mv.piece_moved[1] == "K" or (mv.end_row, mv.end_col) in valid_squares]
            else:
                # Double check: two pieces are giving check, so the king must move (no other piece can block two lines of attack at once)
                moves = []
                self._get_king_moves(king_row, king_col, moves)
        else:
            # Not in check: all pseudo-legal moves are candidates
            moves = self._get_all_possible_moves()
            # Include castling moves if allowed and safe
            self._get_castle_moves(king_row, king_col, moves)

        valids: list[Move] = []
        for move in moves:
            if move.piece_moved[1] == "K":
                if  move.piece_moved[0] == "w":
                    if abs(move.end_row - self.black_king_location[0]) + abs(move.end_col - self.black_king_location[1]) <= 2 and abs(move.end_row - self.black_king_location[0]) != 2 and abs(move.end_col - self.black_king_location[1]) !=2:
                        continue
                else:
                    if abs(move.end_row - self.white_king_location[0]) + abs(move.end_col - self.white_king_location[1]) <= 2 and abs(move.end_row - self.white_king_location[0]) != 2 and abs(move.end_col - self.white_king_location[1]) !=2:
                        continue
            valids.append(move)

        # Restore castling rights (just in case generating moves temporarily changed them)
        self.current_castling_rights = temp_castle_rights.copy()
        return valids

    def _get_all_possible_moves(self) -> list["Move"]:
        """
        Generate all pseudo-legal moves for the current side to move, *without* considering checks to the king.
        This respects piece movement rules and board boundaries, and also accounts for pins (pinned pieces moves are restricted).
        The moves returned here may leave the king in check, so they need to be filtered if used directly.
        """
        moves: list[Move] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                if (piece[0] == "w" and self.white_to_move) or (piece[0] == "b" and not self.white_to_move):
                    # It's a piece of the side to move; generate its moves
                    piece_type = piece[1]
                    if piece_type == "P":
                        self._get_pawn_moves(r, c, moves)
                    elif piece_type == "R":
                        self._get_rook_moves(r, c, moves)
                    elif piece_type == "N":
                        self._get_knight_moves(r, c, moves)
                    elif piece_type == "B":
                        self._get_bishop_moves(r, c, moves)
                    elif piece_type == "Q":
                        self._get_queen_moves(r, c, moves)
                    elif piece_type == "K":
                        self.get_king_moves(r, c, moves)
        return moves

    def _get_pawn_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible pawn moves for the pawn at (row, col), including advances, captures, and en passant.
        Takes into account pin restrictions: a pinned pawn can only move along the pin direction (if at all).
        """
        piece_color = self.board[row][col][0]  # 'w' or 'b'
        move_dir = -1 if piece_color == 'w' else 1   # White pawns move up (-1), black pawns move down (+1)
        start_row = 6 if piece_color == 'w' else 1   # pawns start at row 6 for white, row 1 for black
        enemy_color = 'b' if piece_color == 'w' else 'w'

        # Check if this pawn is pinned, and if so, in which direction.
        piece_pinned = False
        pin_dir = (0, 0)
        for pin in self.pins.copy():
            if pin[0] == row and pin[1] == col:
                piece_pinned = True
                pin_dir = (pin[2], pin[3])
                # Remove the pin entry for this piece to avoid considering it again in other move generation
                self.pins.remove(pin)
                break

        # 1. Pawn forward move (one square)
        if 0 <= row + move_dir < 8:
            if self.board[row + move_dir][col] == EMPTY_SQUARE:
                if not piece_pinned or pin_dir == (move_dir, 0):
                    # Single step forward
                    moves.append(Move((row, col), (row + move_dir, col), self.board))
                    # Double step forward from starting rank (if both squares are free)
                    if row == start_row and self.board[row + 2 * move_dir][col] == EMPTY_SQUARE:
                        moves.append(Move((row, col), (row + 2 * move_dir, col), self.board))
        # 2. Pawn captures (diagonals)
        for dc in (-1, 1):
            new_row = row + move_dir
            new_col = col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if not piece_pinned or (pin_dir == (move_dir, dc)):
                    # Capture enemy piece diagonally
                    if self.board[new_row][new_col][0] == enemy_color:
                        moves.append(Move((row, col), (new_row, new_col), self.board))
                    # En passant capture
                    if self.en_passant_target:
                        # If the last move was a double-step pawn move that landed immediately next to this pawn
                        if (self.en_passant_target.end_row == row and 
                            self.en_passant_target.end_col == new_col and 
                            abs(self.en_passant_target.end_col - col) == 1):
                            # Perform an additional check: ensure en passant capture does not expose our king to check
                            king_row, king_col = (self.white_king_location if piece_color == 'w' else self.black_king_location)
                            attacking_piece = False
                            blocking_piece = False
                            if king_row == row:  # The pawn and king are on the same rank
                                if king_col < col:
                                    # King is to the left of pawn
                                    inside_range = range(king_col + 1, col)       # squares between king and pawn (exclusive)
                                    outside_range = range(col + 1, 8)            # squares to the right of pawn
                                else:
                                    # King is to the right of pawn
                                    inside_range = range(king_col - 1, col, -1)  # squares between king and pawn
                                    outside_range = range(col - 1, -1, -1)       # squares to the left of pawn
                                # Check if any piece blocks the line between king and pawn
                                for c2 in inside_range:
                                    if self.board[row][c2] != EMPTY_SQUARE:
                                        blocking_piece = True
                                        break
                                # If no blocking piece between king and pawn, check if an enemy rook/queen attacks along that line after pawn moves
                                if not blocking_piece:
                                    for c2 in outside_range:
                                        if self.board[row][c2] != EMPTY_SQUARE:
                                            if self.board[row][c2][0] == enemy_color and self.board[row][c2][1] in ("R", "Q"):
                                                attacking_piece = True
                                            break
                            # Only add en passant move if it does not expose king to attack
                            if not attacking_piece:
                                moves.append(Move((row, col), (new_row, new_col), self.board))

    def _get_rook_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible rook moves for the rook at (row, col) along the four cardinal directions.
        If the rook (or queen acting as rook) is pinned, restrict movement to the pin direction (or its opposite).
        """
        piece_color = self.board[row][col][0]
        enemy_color = 'b' if piece_color == 'w' else 'w'
        # Check pin status for this piece
        piece_pinned = False
        pin_dir = (0, 0)
        for pin in self.pins.copy():
            if pin[0] == row and pin[1] == col:
                piece_pinned = True
                pin_dir = (pin[2], pin[3])
                # If this piece is not a queen, remove pin after processing (queen can have a diagonal pin too, handle later)
                if self.board[row][col][1] != "Q":
                    self.pins.remove(pin)
                break

        # Directions for rooks: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            if piece_pinned and not (pin_dir == (dr, dc) or pin_dir == (-dr, -dc)):
                continue  # If pinned and not moving along the pin line, skip this direction
            for i in range(1, BOARD_SIZE):
                new_r = row + dr * i
                new_c = col + dc * i
                if 0 <= new_r < 8 and 0 <= new_c < 8:
                    if self.board[new_r][new_c] == EMPTY_SQUARE:
                        moves.append(Move((row, col), (new_r, new_c), self.board))
                    elif self.board[new_r][new_c][0] == enemy_color:
                        moves.append(Move((row, col), (new_r, new_c), self.board))
                        break  # cannot move past capturing an enemy piece
                    else:
                        break  # blocked by own piece
                else:
                    break  # off board

    def _get_knight_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible knight moves for the knight at (row, col).
        Knights move in 'L' shapes. If a knight is pinned, it cannot move (because it can't move along the pin line).
        """
        # If this knight is pinned, it cannot move at all (any knight move would leave the line between king and attacker).
        for pin in self.pins.copy():
            if pin[0] == row and pin[1] == col:
                # A pinned knight has no legal moves (skip generating moves for this knight)
                return

        knight_moves = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, -1), (2, 1), (-1, -2), (1, -2)]
        ally_color = self.board[row][col][0]
        for dr, dc in knight_moves:
            new_r = row + dr
            new_c = col + dc
            if 0 <= new_r < 8 and 0 <= new_c < 8:
                # Knight can move to the square if it's not occupied by an ally piece
                if self.board[new_r][new_c][0] != ally_color:
                    moves.append(Move((row, col), (new_r, new_c), self.board))

    def _get_bishop_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible bishop moves for the bishop at (row, col) along diagonal directions.
        If the bishop (or queen acting as a bishop) is pinned, restrict movement to the pin direction (or its opposite).
        """
        piece_color = self.board[row][col][0]
        enemy_color = 'b' if piece_color == 'w' else 'w'
        piece_pinned = False
        pin_dir = (0, 0)
        for pin in self.pins.copy():
            if pin[0] == row and pin[1] == col:
                piece_pinned = True
                pin_dir = (pin[2], pin[3])
                # Remove pin from list once found (safe because a piece can only be pinned in one direction at a time)
                self.pins.remove(pin)
                break

        # Directions for bishops: four diagonals
        directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if piece_pinned and not (pin_dir == (dr, dc) or pin_dir == (-dr, -dc)):
                continue  # skip directions that are not along the pin line
            for i in range(1, BOARD_SIZE):
                new_r = row + dr * i
                new_c = col + dc * i
                if 0 <= new_r < 8 and 0 <= new_c < 8:
                    if self.board[new_r][new_c] == EMPTY_SQUARE:
                        moves.append(Move((row, col), (new_r, new_c), self.board))
                    elif self.board[new_r][new_c][0] == enemy_color:
                        moves.append(Move((row, col), (new_r, new_c), self.board))
                        break
                    else:
                        break
                else:
                    break

    def _get_queen_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible queen moves for the queen at (row, col).
        The queen's moves are a combination of rook and bishop moves.
        Pins are handled in the respective rook/bishop move generation functions.
        """
        # Generate moves in straight lines (like a rook) and diagonals (like a bishop)
        self._get_rook_moves(row, col, moves)
        self._get_bishop_moves(row, col, moves)

    def get_king_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Get all possible king moves for the king at (row, col) - one square in any direction.
        (This does not automatically filter out moves into check; those are handled when validating legal moves.)
        """
        row_moves = (-1, -1, -1, 0, 0, 1, 1, 1)
        col_moves = (-1, 0, 1, -1, 1, -1, 0, 1)
        ally_color = "w" if self.white_to_move else "b"
        for i in range(8):
            end_row = row + row_moves[i]
            end_col = col + col_moves[i]
            if 0 <= end_row <= 7 and 0 <= end_col <= 7:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] != ally_color:  # not an ally piece - empty or enemy
                    # place king on end square and check for checks
                    if ally_color == "w":
                        self.white_king_location = (end_row, end_col)
                    else:
                        self.black_king_location = (end_row, end_col)
                    in_check, pins, checks = self._check_for_pins_and_checks()
                    if not in_check:
                        moves.append(Move((row, col), (end_row, end_col), self.board))
                    # place king back on original location
                    if ally_color == "w":
                        self.white_king_location = (row, col)
                    else:
                        self.black_king_location = (row, col)


    def _get_king_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        # Internal helper to generate king moves (possibly used during move filtering)
        self.get_king_moves(row, col, moves)

    def _check_for_pins_and_checks(self) -> tuple[bool, list[tuple[int,int,int,int]], list[tuple[int,int,int,int]]]:
        """
        Look for any pins or checks on the king of the side to move.
        Returns:
            in_check (bool): True if the current player's king is in check.
            pins (list): List of pinned pieces (their coordinates and the direction of the pin).
            checks (list): List of checking pieces (their coordinates and direction toward the king).
        """
        pins = []
        checks = []
        in_check = False

        if self.white_to_move:
            ally_color = "w"
            enemy_color = "b"
            start_row, start_col = self.white_king_location
        else:
            ally_color = "b"
            enemy_color = "w"
            start_row, start_col = self.black_king_location

        # Directions from the king for rays: up, left, down, right, up-left, up-right, down-left, down-right
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for j, (dr, dc) in enumerate(directions):
            possible_pin: tuple[int, int, int, int] | None = None
            for i in range(1, BOARD_SIZE):
                end_r = start_row + dr * i
                end_c = start_col + dc * i
                if 0 <= end_r < 8 and 0 <= end_c < 8:
                    piece = self.board[end_r][end_c]
                    if piece[0] == ally_color and piece[1] != "K":
                        # First allied piece along this direction could be pinned by an enemy behind it
                        if possible_pin is None:
                            possible_pin = (end_r, end_c, dr, dc)
                        else:
                            # Second allied piece found in this direction means no pin or check possible along this line
                            break
                    elif piece[0] == enemy_color:
                        piece_type = piece[1]
                        # Check if the enemy piece is a rook, bishop or queen along appropriate direction, or a pawn/knight close by
                        if (j <= 3 and piece_type == "R") or (j >= 4 and piece_type == "B") or piece_type == "Q":
                            # Enemy rook/queen along straight line or bishop/queen along diagonal can give check/pin
                            if possible_pin is None:
                                # No blocking piece: this is a direct check to the king
                                in_check = True
                                checks.append((end_r, end_c, dr, dc))
                            else:
                                # An allied piece is between king and this enemy: that ally is pinned
                                pins.append(possible_pin)
                            break  # break out after handling check/pin
                        if i == 1 and piece_type == "P":
                            # Enemy pawn can check the king if it is one square away diagonally:
                            # For white king (enemy pawn is black): black pawn checks from down-left or down-right (j == 6 or 7)
                            # For black king (enemy pawn is white): white pawn checks from up-left or up-right (j == 4 or 5)
                            if (enemy_color == "w" and 6 <= j <= 7) or (enemy_color == "b" and 4 <= j <= 5):
                                if possible_pin is None:
                                    in_check = True
                                    checks.append((end_r, end_c, dr, dc))
                        # If enemy piece is not a sliding piece giving check in line, then this direction is blocked (e.g., knight or king or irrelevant piece)
                        break
                    else:
                        # Empty square; continue scanning further along this direction
                        continue
                else:
                    break  # off board, stop scanning this direction

        # Check for knight checks (knights move in L-shapes which are not covered by the above ray directions)
        knight_moves = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, -1), (2, 1), (-1, -2), (1, -2)]
        for dr, dc in knight_moves:
            end_r = start_row + dr
            end_c = start_col + dc
            if 0 <= end_r < 8 and 0 <= end_c < 8:
                piece = self.board[end_r][end_c]
                if piece[0] == enemy_color and piece[1] == "N":
                    # Enemy knight can attack the king
                    in_check = True
                    checks.append((end_r, end_c, dr, dc))
                    
        return in_check, pins, checks

    def _square_under_attack(self, r: int, c: int) -> bool:
        """
        Determine if the square (r, c) is under attack by any piece of the opponent.
        This is used primarily to verify that squares the king crosses during castling are safe.
        """
        current_side = self.white_to_move
        # Temporarily set to opponent's turn to generate their moves
        self.white_to_move = not current_side
        opponent_moves = self._get_all_possible_moves()
        self.white_to_move = current_side
        for mv in opponent_moves:
            if mv.end_row == r and mv.end_col == c:
                return True
        return False

    def _get_castle_moves(self, row: int, col: int, moves: list["Move"]) -> None:
        """
        Generate all valid castling moves for the king at (row, col) and add them to the moves list.
        Conditions for castling:
         - The king is not currently in check.
         - The squares between the king and the rook are empty.
         - The king does not pass through or end on a square that is under attack.
         - The corresponding castling rights (wks, wqs, bks, bqs) are still True.
        """
        # Do not allow castling if the king is in check.
        if self._square_under_attack(row, col):
            return
        if self.white_to_move:
            # White king castling
            if self.current_castling_rights["wks"]:
                # King-side castling (white king from e1 to g1, rook from h1 to f1)
                if col+2 < 8 and self.board[row][col+1] == EMPTY_SQUARE and self.board[row][col+2] == EMPTY_SQUARE:
                    if not self._square_under_attack(row, col+1) and not self._square_under_attack(row, col+2):
                        moves.append(Move((row, col), (row, col+2), self.board))
            if self.current_castling_rights["wqs"]:
                # Queen-side castling (white king from e1 to c1, rook from a1 to d1)
                if (col >= 3 and self.board[row][col-1] == EMPTY_SQUARE and 
                    self.board[row][col-2] == EMPTY_SQUARE and 
                    self.board[row][col-3] == EMPTY_SQUARE):
                    if not self._square_under_attack(row, col-1) and not self._square_under_attack(row, col-2):
                        moves.append(Move((row, col), (row, col-2), self.board))
        else:
            # Black king castling
            if self.current_castling_rights["bks"]:
                # King-side castling (black king from e8 to g8, rook from h8 to f8)
                if col+2 < 8 and self.board[row][col+1] == EMPTY_SQUARE and self.board[row][col+2] == EMPTY_SQUARE:
                    if not self._square_under_attack(row, col+1) and not self._square_under_attack(row, col+2):
                        moves.append(Move((row, col), (row, col+2), self.board))
            if self.current_castling_rights["bqs"]:
                # Queen-side castling (black king from e8 to c8, rook from a8 to d8)
                if (col >= 3 and self.board[row][col-1] == EMPTY_SQUARE and 
                    self.board[row][col-2] == EMPTY_SQUARE and 
                    self.board[row][col-3] == EMPTY_SQUARE):
                    if not self._square_under_attack(row, col-1) and not self._square_under_attack(row, col-2):
                        moves.append(Move((row, col), (row, col-2), self.board))

    # ------------------------------------------------------------------#
    # State Evaluation and Check Detection
    # ------------------------------------------------------------------#
    def in_check(self, player: str) -> bool:
        """
        Check if the given player's king is in check.
        :param player: 'w' for white, 'b' for black.
        :return: True if that player's king is under attack by any of the opponent's pieces.
        """
        # Determine which king's position we are evaluating
        king_position = self.white_king_location if player == "w" else self.black_king_location
        # Generate all moves for the opponent and see if any move can land on the king's position
        current_side = self.white_to_move
        # Flip perspective to opponent
        self.white_to_move = (player != "w")
        opponent_moves = self._get_all_possible_moves()
        self.white_to_move = current_side
        for mv in opponent_moves:
            if mv.end_row == king_position[0] and mv.end_col == king_position[1]:
                return True
        return False

    def draw_game(self) -> None:
        """
        Check and update draw conditions:
         - Fifty-move rule (no pawn moves or captures in the last 50 half-moves).
         - Insufficient material (not enough pieces to force a checkmate).
         - Stalemate (handled separately by stalemate flag, but also triggers draw).
         - (Threefold repetition could be added here.)
        If any condition is met, set self.draw = True.
        """
        # 50-move rule: if 50 half-moves (ply) have passed since the last pawn move or capture, it's a draw.
        if ((self.last_capture_turn != -1 and self.turn - self.last_capture_turn >= 50) or 
            (self.last_pawn_turn != -1 and self.turn - self.last_pawn_turn >= 50)):
            self.fifty_rule = True
            self.draw = True
            return

        # Insufficient material: e.g., King vs King, King vs King+Knight, King vs King+Bishop
        pieces = [sq for row in self.board for sq in row if sq != EMPTY_SQUARE]
        # Sort pieces by type for easier checking
        piece_types = sorted([p[1] for p in pieces])
        # Only kings left
        if piece_types == ["K", "K"]:
            self.insuf_mat = True
            self.draw = True
            return
        # King and one minor piece vs King
        if len(piece_types) == 3 and piece_types.count("B") + piece_types.count("N") == 1 and piece_types.count("K") == 2:
            # This covers K+B vs K or K+N vs K (which are draws due to insufficient mating material)
            self.insuf_mat = True
            self.draw = True
            return
        
        # Threefold repetition
        for count in self.positions_count.values():
            if count >= 3:
                self.draw = True
                self.threefold = True
                return

# ---------------------------------------------------------------------#
# Move class – represents a single move (half-move) in chess
# ---------------------------------------------------------------------#
class Move:
    """
    Immutable object describing a single chess move (half-move).
    It stores the start and end coordinates, the piece moved, and the piece captured (if any),
    as well as flags for special moves like castling, promotion, and en passant.
    """
    # Class-level lookup tables to convert between board coordinates and algebraic notation
    ranks_to_rows = {str(r): 7 - (int(r) - 1) for r in range(1, 9)}  # '1'->7, '2'->6, ..., '8'->0
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {f: i for i, f in enumerate("abcdefgh")}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_sq: tuple[int,int], end_sq: tuple[int,int], board: np.ndarray):
        self.start_row, self.start_col = start_sq
        self.end_row, self.end_col = end_sq

        self.piece_moved = board[self.start_row][self.start_col]    # piece that is moving
        self.piece_captured = board[self.end_row][self.end_col]     # piece that gets captured (if any)

        # Special move flags:
        self.en_passant = self._is_en_passant(board)
        self.promotion = self._is_promotion()
        self.castling = self._is_castling()

    def _is_en_passant(self, board: np.ndarray) -> bool:
        """
        Determine if this move is an en passant capture.
        It's en passant if a pawn moves diagonally into an empty square, and the piece captured is a pawn 
        located adjacent to the start square (the pawn that moved two steps in the previous move).
        """
        # Only pawns can perform en passant
        if self.piece_moved[1] != "P":
            return False
        # En passant move: pawn moves diagonally into a square that appears empty, and captures an adjacent pawn.
        if abs(self.end_col - self.start_col) == 1 and self.piece_captured == EMPTY_SQUARE:
            # The pawn to be captured is not on the end square, but on the side of the start square.
            victim_color = "b" if self.piece_moved[0] == "w" else "w"
            # Check the square where the opposing pawn would have been (same row as start, column = end_col)
            return board[self.start_row][self.end_col] == victim_color + "P"
        return False

    def _is_promotion(self) -> bool:
        """
        Determine if this move is a pawn promotion.
        True if a pawn moved to the last rank (row 0 for white pawn or row 7 for black pawn).
        """
        return (self.piece_moved[1] == "P" and 
                (self.end_row == 0 or self.end_row == 7))

    def _is_castling(self) -> bool:
        """
        Determine if this move is a castling move.
        True if a king moved two squares horizontally (which is how castling is represented in this engine).
        """
        return (self.piece_moved[1] == "K" and abs(self.start_col - self.end_col) == 2)

    def __eq__(self, other: object) -> bool:
        """Override equality to allow Move comparisons based on start and end coordinates."""
        if not isinstance(other, Move):
            return False
        return (self.start_row == other.start_row and self.start_col == other.start_col and 
                self.end_row == other.end_row and self.end_col == other.end_col)

    def __hash__(self) -> int:
        # This allows Move objects to be used in sets or as dictionary keys.
        return hash((self.start_row, self.start_col, self.end_row, self.end_col))

    def get_chess_notation(self) -> str:
        """
        Return the move in standard algebraic notation (e.g., 'e2e4', 'e7e8Q' for promotion, '0-0' for king-side castle).
        This is a basic notation without disambiguation or check/mate symbols.
        """
        if self.castling:
            return "0-0" if self.end_col == 6 else "0-0-0"
        promotion_suffix = "Q" if self.promotion and self.piece_moved[1] == "P" else ""
        return self._algebraic(self.start_row, self.start_col) + self._algebraic(self.end_row, self.end_col) + promotion_suffix

    def _algebraic(self, row: int, col: int) -> str:
        # Helper to convert board coordinates to algebraic notation (e.g., (7,0) -> 'a1')
        return self.cols_to_files[col] + self.rows_to_ranks[row]
