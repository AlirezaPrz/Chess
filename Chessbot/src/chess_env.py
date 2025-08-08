# ChessEnv.py (Improved Custom Environment for Chess RL)
import gym
from gym.spaces import Discrete, Box
import numpy as np
from chess_engine import GameState, Move

# Piece values for reward calculation (for captured pieces)
_piece_values = {
    'P': 1, 'p': 1,   # pawn (we use 'P' for white pawn, 'p' for black pawn if needed)
    'N': 3, 'n': 3,   # knight
    'B': 3, 'b': 3,   # bishop
    'R': 5, 'r': 5,   # rook
    'Q': 9, 'q': 9,   # queen
    'K': 0, 'k': 0    # king (king's capture ends game; handled via checkmate reward)
}

class ChessEnv(gym.Env):
    """
    Custom Gym environment for Chess using the ChessEngine GameState.
    Observation: 65-dimensional numpy array (64 squares + 1 turn indicator).
    Action: Discrete(4096) representing from_square*64 + to_square.
    Reward: +value for capturing a piece, +100 for delivering checkmate, 0 for draw or illegal move.
    The agent will play both sides (self-play) by taking alternate turns.
    """
    def __init__(self):
        super().__init__()
        # 4096 possible actions (including many that will be illegal in most positions)
        self.action_space = Discrete(4096)
        # Observation: 64 board squares + 1 turn indicator. Values in [-6,6].
        self.observation_space = Box(low=-6, high=6, shape=(65,), dtype=np.int32)
        # Initialize game state
        self.gs = GameState()
    
    def _encode_state(self):
        """Encode the current GameState into a 65-length integer vector."""
        state = np.zeros(65, dtype=np.int32)
        idx = 0
        for r in range(8):
            for c in range(8):
                piece = self.gs.board[r][c]
                code = 0
                if piece != "--":  # not an empty square
                    piece_type = piece[1]  # e.g., "P", "R", "N", etc.
                    # Assign base value by piece type
                    if piece_type == 'P': val = 1
                    elif piece_type == 'N': val = 2
                    elif piece_type == 'B': val = 3
                    elif piece_type == 'R': val = 4
                    elif piece_type == 'Q': val = 5
                    elif piece_type == 'K': val = 6
                    else: val = 0
                    # Use positive for White pieces, negative for Black pieces
                    code = val if piece[0] == 'w' else -val
                state[idx] = code
                idx += 1
        # Turn indicator: 1 if White to move, 0 if Black to move
        state[64] = 1 if self.gs.white_to_move else 0
        return state
    
    def reset(self):
        """Reset the environment to the initial chess position. Returns initial observation."""
        self.gs = GameState()  # new game state (standard initial chess setup)
        return self._encode_state()
    
    def step(self, action):
        """
        Apply the given action (move) to the game. Action is an int 0-4095.
        Returns (observation, reward, done, info).
        """
        # Decode the action index into source and target squares
        from_sq = action // 64
        to_sq = action % 64
        from_row, from_col = divmod(from_sq, 8)
        to_row, to_col = divmod(to_sq, 8)
        move = Move((from_row, from_col), (to_row, to_col), self.gs.board)
        # If the move is not legal, replace it with a random legal move (if available)
        legal_moves = self.gs.get_valid_moves()
        legal_actions = []
        for m in legal_moves:
            start_idx = m.start_row * 8 + m.start_col
            end_idx = m.end_row * 8 + m.end_col
            legal_actions.append(start_idx * 64 + end_idx)
        if action not in legal_actions:
            if legal_actions:
                # Choose a random legal move (could also choose first or skip)
                action = np.random.choice(legal_actions)
                from_sq = action // 64
                to_sq = action % 64
                from_row, from_col = divmod(from_sq, 8)
                to_row, to_col = divmod(to_sq, 8)
                move = Move((from_row, from_col), (to_row, to_col), self.gs.board)
            # If no legal moves, the game is over (checkmate or stalemate)
        # Apply the move
        self.gs.make_move(move)
        # Calculate reward for this move
        reward = 0
        # If a piece was captured, add its value to reward
        if move.piece_captured != "--":
            piece_char = move.piece_captured[1]  # e.g., 'Q' or 'P'
            # Use lowercase for black captured pieces to lookup in _piece_values
            if piece_char.islower() or piece_char.isupper():
                # Ensure we use correct key (uppercase in our dict for white, lowercase for black)
                key = piece_char  # already 'P','Q' etc. If the piece was black, it's in lowercase in board notation?
                reward += _piece_values.get(key, 0)
        # Check for game over conditions
        done = False
        if self.gs.checkmate:
            done = True
            # +100 reward for delivering checkmate (winning move)
            reward += 100
        elif self.gs.stalemate or self.gs.draw:
            done = True
            # No additional reward for draw (could be set small if desired)
            reward += 0
        # Get next state observation
        obs = self._encode_state()
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        """Print a simple text-based board. Useful for debugging."""
        symbols = {
            "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
            "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟", "--": "."
        }
        for r in range(8):
            row_str = ""
            for c in range(8):
                piece = self.gs.board[r][c]
                row_str += symbols.get(piece, ".") + " "
            print(row_str)
        print("Turn:", "White" if self.gs.white_to_move else "Black")
