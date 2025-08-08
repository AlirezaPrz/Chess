"""
Main driver file.
Handles:
    • Menu navigation (start, mode select, bot difficulty)
    • Rendering / input for menus
    • Hand off to the existing PvP game loop
    • Utility routines: board drawing, animation, etc.
"""

import chess_engine
import pygame as p
import button                     # custom button class
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import math
from minimax_bot import evaluate_board, minimax, find_best_move

# ------------------------------------------------------------------ #
# CONSTANTS & GLOBALS                                                #
# ------------------------------------------------------------------ #
p.init()
WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}
COLORS = [p.Color("navajowhite"), p.Color("saddlebrown")]

# Global variable to hold the loaded DQN model for the easy bot
bot_model = None

# ------------------------------------------------------------------ #
# ASSET LOADING                                                      #
# ------------------------------------------------------------------ #
def load_images() -> None:
    """Load piece sprites + menu graphics into the global IMAGES dict."""
    pieces = [
        "bR", "bN", "bB", "bQ", "bK", "bP",
        "wR", "wN", "wB", "wQ", "wK", "wP",
    ]
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(
            p.image.load(f"../images/set2/{piece}.png"), (SQ_SIZE, SQ_SIZE)
        )

    # Menu graphics
    IMAGES["background"] = p.transform.scale(
        p.image.load("../images/start/background.JPEG"), (WIDTH, HEIGHT)
    )
    # buttons: start, exit, pvp, bots, easy, medium, hard
    for btn in ["pvp", "bots", "easy", "medium", "hard"]:
        IMAGES[btn] = p.image.load(f"../images/start/{btn}_btn.jpg")  # raw size
    for btn in ["start", "exit"]:
        IMAGES[btn] = p.image.load(f"../images/start/{btn}_btn.png")  # raw size
        
def get_bot_move(gs: chess_engine.GameState, difficulty: str) -> chess_engine.Move:
    """
    Determine the bot's move based on the difficulty level.
    - Medium/Hard: use minimax search.
    - Easy: use the trained model if available, otherwise a random move.
    """
    if difficulty in ("medium", "hard"):
        # Set deeper search for hard difficulty
        # End game
        if gs.num_pieces <= 4:
            search_depth = 7 if difficulty == "medium" else 10
        elif gs.num_pieces <= 5:
            search_depth = 6 if difficulty == "medium" else 9
        elif gs.num_pieces <= 6:
            search_depth = 5 if difficulty == "medium" else 8
        elif gs.num_pieces <= 8:
            search_depth = 4 if difficulty == "medium" else 6
        # Opening and mid-game
        else:
            search_depth = 3 if difficulty == "medium" else 4
        return find_best_move(gs, search_depth)
    elif difficulty == "easy":
        if bot_model is not None:
            # Use the loaded DQN model to pick a move
            # Encode the current game state as a 65-length vector (64 squares + side-to-move)
            state_vec = []
            for r in range(8):
                for c in range(8):
                    piece = gs.board[r][c]
                    code = 0
                    if piece != chess_engine.EMPTY_SQUARE:
                        # Numeric encoding similar to training (pawn=1, knight=2, bishop=3, rook=4, queen=5, king=6)
                        piece_type = piece[1]
                        if piece_type == 'P': val = 1
                        elif piece_type == 'N': val = 2
                        elif piece_type == 'B': val = 3
                        elif piece_type == 'R': val = 4
                        elif piece_type == 'Q': val = 5
                        elif piece_type == 'K': val = 6
                        else: val = 0
                        code = val if piece[0] == 'w' else -val
                    state_vec.append(code)
            state_vec.append(1 if gs.white_to_move else 0)
            state_array = np.array(state_vec, dtype=np.int32).reshape(1, -1)
            q_values = bot_model.predict(state_array, verbose=0)[0]
            # Select the legal move with the highest predicted Q-value
            legal_moves = gs.get_valid_moves()
            best_move = None
            best_q = -float('inf')
            for m in legal_moves:
                start_idx = m.start_row * 8 + m.start_col
                end_idx   = m.end_row * 8 + m.end_col
                action_index = start_idx * 64 + end_idx
                if q_values[action_index] > best_q:
                    best_q = q_values[action_index]
                    best_move = m
            return best_move
        else:
            # No model loaded, choose a random legal move
            moves = gs.get_valid_moves()
            return random.choice(moves) if moves else None


# ------------------------------------------------------------------ #
# MAIN                                                              #
# ------------------------------------------------------------------ #
def main() -> None:
    """Entry-point for the program."""
    load_images()

    screen = p.display.set_mode((WIDTH, HEIGHT))
    p.display.set_caption("Chess")
    clock = p.time.Clock()

    # ------------------ MENU SETUP ------------------ #
    # Buttons are created **once** and re‑drawn each frame.
    start_btn  = button.button(175, 140, IMAGES["start"], 0.65)
    exit_btn   = button.button(185, 300, IMAGES["exit"], 0.65)

    pvp_btn    = button.button(56, 200, IMAGES["pvp"], 0.16)
    bots_btn   = button.button(296, 200, IMAGES["bots"], 0.16)

    easy_btn   = button.button(175, 100, IMAGES["easy"], 0.16)
    med_btn    = button.button(175, 220, IMAGES["medium"], 0.16)
    hard_btn   = button.button(175, 340, IMAGES["hard"], 0.16)

    # ------------------ GAME STATE ------------------ #
    global bot_model
    gs = chess_engine.GameState()
    valid_moves = gs.get_valid_moves()
    sq_selected: list[tuple[int, int]] = []
    game_over = False

    # Finite‑state‑machine variable
    screen_state = "START_MENU"       # START_MENU → MODE_MENU → BOT_MENU → GAME
                                      #                        → PVP_MENU

    bot_level = None                  # "easy" | "medium" | "hard" (placeholder)

    running = True
    while running:
        # ---------------------------------------------------- #
        # 1. Handle window‑level events (quit)                 #
        # ---------------------------------------------------- #
        events = p.event.get()
        for event in events:
            if event.type == p.QUIT:
                running = False

        # ---------------------------------------------------- #
        # 2. Screen‑specific logic & rendering                 #
        # ---------------------------------------------------- #
        if screen_state == "START_MENU":
            screen.blit(IMAGES["background"], (0, 0))
            if start_btn.draw(screen):
                screen_state = "MODE_MENU"
            if exit_btn.draw(screen):
                running = False

        elif screen_state == "MODE_MENU":
            screen.blit(IMAGES["background"], (0, 0))
            if pvp_btn.draw(screen):
                screen_state = "PVP_MENU"       # go straight to PvP
            if bots_btn.draw(screen):
                screen_state = "BOT_MENU"   # choose difficulty

        elif screen_state == "BOT_MENU":
            screen.blit(IMAGES["background"], (0, 0))
            if easy_btn.draw(screen):
                bot_level = "easy"
                screen_state = "GAME"
            if med_btn.draw(screen):
                bot_level = "medium"
                screen_state = "GAME"
            if hard_btn.draw(screen):
                bot_level = "hard"
                screen_state = "GAME"

        elif screen_state == "PVP_MENU":
            #player vs player mode
            for event in events:
                if event.type == p.QUIT:
                    running = False
                # Handle other events like mouse clicks, key presses, etc. here
                elif event.type == p.MOUSEBUTTONDOWN:
                    if not game_over:
                        location = p.mouse.get_pos()
                        col = location[0] // SQ_SIZE
                        row = location[1] // SQ_SIZE

                        if sq_selected == [(row, col)]:  # Deselect the square
                            sq_selected = []
                        else:
                            sq_selected.append((row, col))  # Select the square
                        if len(sq_selected) == 2:
                            # Handle the move logic here
                            move = chess_engine.Move(sq_selected[0], sq_selected[1], gs.board)
                            print(move.get_chess_notation())
                            if move in valid_moves:
                                gs.make_move(move)  # Make the move and update the game state
                                if gs.checkmate or gs.stalemate or gs.draw:
                                    game_over = True
                                valid_moves = gs.get_valid_moves()  # Update valid moves after the move
                                sq_selected = []  # Reset selection after making a move
                                animate_move(move, screen, gs.board, clock)  # Animate the move
                            else:
                                sq_selected = [(row, col)]  # Invalid move, keep the selection
                elif event.type == p.KEYDOWN:
                    if event.key == p.K_z and not game_over:
                        gs.undo_move()
                        valid_moves = gs.get_valid_moves()  # Update valid moves after undo
                        sq_selected = []
                    if event.key == p.K_r:
                        gs = chess_engine.GameState()
                        valid_moves = gs.get_valid_moves()  # Reset valid moves after reset
                        sq_selected = []
                        game_over = False  # Reset game over state
                        
                    if event.key == p.K_q:
                        gs = chess_engine.GameState()
                        valid_moves = gs.get_valid_moves()  # Reset valid moves after reset
                        sq_selected = []
                        game_over = False  # Reset game over state
                        screen_state = "START_MENU"
                        

            draw_game_state(screen, gs, valid_moves, sq_selected[0] if sq_selected else ())  # Draw the game state
            if game_over:
                if gs.checkmate:
                    if gs.winner == "b":
                        draw_text(screen, "Black wins by checkmate!")
                    else:
                        draw_text(screen, "White wins by checkmate!")
                elif gs.stalemate:
                    draw_text(screen, "Stalemate!")
                elif gs.draw:
                    if gs.threefold:
                        draw_text(screen, "Draw by threefold rule!")
                    elif gs.fifty_rule:
                        draw_text(screen, "Draw by 50-move rule!")
                    else: 
                        draw_text(screen, "Draw by insufficient material!")
            clock.tick(MAX_FPS)
            p.display.flip()
            
        elif screen_state == "GAME":
            # If just entering the GAME state, load the model for easy difficulty (medium/hard skip model)
            if bot_model is None:
                if bot_level in ("medium", "hard"):
                    bot_model = None  # no model needed for minimax
                else:
                    try:
                        bot_model = tf.keras.models.load_model(f"weights/hard_dqn_model_tf.h5", compile=False)
                    except Exception as e:
                        print(f"Error loading model for {bot_level}: {e}")
                        bot_model = None

            # Handle player (White) input events
            for event in events:
                if event.type == p.QUIT:
                    running = False
                elif event.type == p.MOUSEBUTTONDOWN:
                    if not game_over and gs.white_to_move:
                        x, y = p.mouse.get_pos()
                        col = x // SQ_SIZE
                        row = y // SQ_SIZE
                        if sq_selected == [(row, col)]:
                            sq_selected = []  # deselect if same square clicked again
                        else:
                            sq_selected.append((row, col))
                        if len(sq_selected) == 2:
                            move = chess_engine.Move(sq_selected[0], sq_selected[1], gs.board)
                            if move in valid_moves:
                                gs.make_move(move)
                                animate_move(move, screen, gs.board, clock)
                                if gs.checkmate or gs.stalemate or gs.draw:
                                    game_over = True
                                else:
                                    valid_moves = gs.get_valid_moves()
                                sq_selected = []
                            else:
                                # Invalid move, reset selection to the second square
                                sq_selected = [(row, col)]
                elif event.type == p.KEYDOWN:
                    if event.key == p.K_z and not game_over:
                        # Undo player's and bot's last moves
                        if gs.move_log:
                            gs.undo_move()
                            if gs.move_log:
                                gs.undo_move()
                        game_over = False
                        valid_moves = gs.get_valid_moves()
                        sq_selected = []
                    if event.key == p.K_r:
                        # Restart the game with the same bot
                        gs = chess_engine.GameState()
                        valid_moves = gs.get_valid_moves()
                        sq_selected = []
                        game_over = False
                    if event.key == p.K_q:
                        # Quit to main menu
                        gs = chess_engine.GameState()
                        valid_moves = gs.get_valid_moves()
                        sq_selected = []
                        game_over = False
                        screen_state = "START_MENU"
                        bot_model = None  # reset model when leaving to menu

            # Bot's turn (Black)
            if not game_over and not gs.white_to_move:
                bot_move = get_bot_move(gs, bot_level)
                if bot_move is None:
                    # No moves (checkmate or stalemate)
                    game_over = True
                else:
                    gs.make_move(bot_move)
                    animate_move(bot_move, screen, gs.board, clock)
                    if gs.checkmate or gs.stalemate or gs.draw:
                        game_over = True
                    else:
                        valid_moves = gs.get_valid_moves()
                # After the bot move, control goes back to White

            # Render the board and pieces
            draw_game_state(screen, gs, valid_moves, sq_selected[0] if sq_selected else ())
            # If game over, display the result message
            if game_over:
                if gs.checkmate:
                    if gs.winner == "b":
                        draw_text(screen, "Black wins by checkmate!")
                    else:
                        draw_text(screen, "White wins by checkmate!")
                elif gs.stalemate:
                    draw_text(screen, "Stalemate!")
                elif gs.draw:
                    if gs.threefold:
                        draw_text(screen, "Draw by threefold repetition!")
                    elif gs.fifty_rule:
                        draw_text(screen, "Draw by 50-move rule!")
                    else:
                        draw_text(screen, "Draw by insufficient material!")

        # Unknown state fallback
        else:
            screen_state = "START_MENU"

        p.display.flip()
        clock.tick(MAX_FPS)


# ------------------------------------------------------------------ #
#  UTILITY FUNCTIONS (unchanged)                                     #
# ------------------------------------------------------------------ #
def highlight_square(screen, gs, valid_moves, sq_selected):
    """
    Highlight selected square and its valid moves.
    """
    if sq_selected != ():
        row, col = sq_selected
        if gs.board[row][col][0] == ('w' if gs.white_to_move else 'b'):
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color("blue"))
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
            for move in valid_moves:
                if move.start_row == row and move.start_col == col:
                    s.fill(p.Color("yellow" if gs.board[move.end_row][move.end_col] == "--" else "red"))
                    screen.blit(s, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))


def draw_game_state(screen, gs, valid_moves, sq_selected):
    """Draw board, pieces, highlights."""
    drawboard(screen) # Draw the squares on the board
    highlight_square(screen, gs, valid_moves, sq_selected) # Highlight the selected square and valid moves
    # Highlight the last move made
    if gs.move_log:
        last_move = gs.move_log[-1]
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)
        s.fill(p.Color("green"))
        screen.blit(s, (last_move.end_col * SQ_SIZE, last_move.end_row * SQ_SIZE))
        # Highlight the start square of the last move
        s.fill(p.Color("orange"))
        screen.blit(s, (last_move.start_col * SQ_SIZE, last_move.start_row * SQ_SIZE))
    draw_pieces(screen, gs.board) # Draw the pieces on the board


def drawboard(screen):
    """Color the 8*8 squares."""
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = COLORS[(r + c) % 2]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_pieces(screen, board):
    """Blit every piece in `board` onto `screen`."""
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            if board[r][c] != "--":
                screen.blit(
                    IMAGES[board[r][c]],
                    p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                )


def animate_move(move, screen, board, clock):
    """Piece-slide animation (unchanged)."""
    coords = []
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    frames_per_square = 10
    frame_count = (abs(dR) + abs(dC)) * frames_per_square
    for frame in range(frame_count + 1):
        r, c = (
            move.start_row + dR * frame / frame_count,
            move.start_col + dC * frame / frame_count
        )
        drawboard(screen)  # Redraw the board
        draw_pieces(screen, board)  # Redraw the pieces
        # erase the moved piece from its starting square
        color = COLORS[(move.start_row + move.start_col) % 2]
        s = p.Surface((SQ_SIZE, SQ_SIZE))
        end_square = p.Rect(move.end_col * SQ_SIZE, move.end_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, end_square)
        # Draw captured piece onto rectangle
        if move.piece_captured != "--":
            screen.blit(IMAGES[move.piece_captured], end_square)
        # Draw the moving piece
        screen.blit(IMAGES[move.piece_moved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)


def draw_text(screen, text):
    """Center-screen message (checkmate / draw)."""
    font = p.font.SysFont("Serif", 32, True, False)
    text_object = font.render(text, 0, p.Color("grey"))
    text_location = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH // 2 - text_object.get_width()/2, HEIGHT // 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    # draw shadow
    text_object = font.render(text, 0, p.Color("black"))
    screen.blit(text_object, text_location.move(2, 2))


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
