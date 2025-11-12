"""
Enhanced chess board visualization using Unicode and ANSI colors.

This module provides beautiful terminal rendering of chess positions with:
- Unicode chess pieces (♔♕♖♗♘♙)
- Colored board squares (light/dark)
- Coordinate labels (a-h, 1-8)
- Last move highlighting
- Captured pieces display
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

# Unicode chess pieces
PIECES_UNICODE = {
    chess.PAWN: {'white': '♙', 'black': '♟'},
    chess.KNIGHT: {'white': '♘', 'black': '♞'},
    chess.BISHOP: {'white': '♗', 'black': '♝'},
    chess.ROOK: {'white': '♖', 'black': '♜'},
    chess.QUEEN: {'white': '♕', 'black': '♛'},
    chess.KING: {'white': '♔', 'black': '♚'},
}

# ANSI color codes
class Colors:
    # Square colors
    LIGHT_SQUARE = '\033[48;5;222m'  # Light tan
    DARK_SQUARE = '\033[48;5;94m'     # Dark brown
    HIGHLIGHT = '\033[48;5;186m'      # Yellow highlight for last move

    # Text colors
    WHITE_PIECE = '\033[97m'          # Bright white
    BLACK_PIECE = '\033[30m'          # Black
    COORD = '\033[90m'                # Gray for coordinates

    # Reset
    RESET = '\033[0m'
    BOLD = '\033[1m'


def get_piece_symbol(piece, use_unicode=True):
    """Get the symbol for a chess piece."""
    if not piece:
        return '  '

    if use_unicode:
        color = 'white' if piece.color == chess.WHITE else 'black'
        symbol = PIECES_UNICODE[piece.piece_type][color]
        return f' {symbol}'
    else:
        # Fallback to ASCII
        return f' {str(piece).upper() if piece.color == chess.WHITE else str(piece).lower()}'


def is_light_square(square):
    """Check if a square is a light square."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (rank + file) % 2 == 1


def display_board_large(board, last_move=None, captured_pieces=None, use_unicode=True, use_colors=True):
    """
    Display a chess board in large format with box-drawing borders.

    Args:
        board: chess.Board object
        last_move: chess.Move object to highlight
        captured_pieces: Dict with 'white' and 'black' lists of captured pieces
        use_unicode: Use Unicode pieces (vs ASCII)
        use_colors: Use ANSI colors (vs plain text)

    Returns:
        String representation of the board
    """
    lines = []

    # Determine which squares to highlight
    highlight_squares = set()
    if last_move:
        highlight_squares = {last_move.from_square, last_move.to_square}

    # Top border with file labels
    if use_colors:
        lines.append(f"{Colors.COORD}      a   b   c   d   e   f   g   h{Colors.RESET}")
    else:
        lines.append("      a   b   c   d   e   f   g   h")

    # Top border of board
    lines.append("    ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    # Board ranks (8 to 1)
    for rank in range(7, -1, -1):
        # Rank label
        if use_colors:
            rank_label = f"{Colors.COORD}{rank + 1}{Colors.RESET}"
        else:
            rank_label = f"{rank + 1}"

        line = f"  {rank_label} │"

        # Files (a to h)
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            # Choose background color
            if use_colors:
                if square in highlight_squares:
                    bg_color = Colors.HIGHLIGHT
                elif is_light_square(square):
                    bg_color = Colors.LIGHT_SQUARE
                else:
                    bg_color = Colors.DARK_SQUARE

                # Choose piece color
                if piece:
                    if piece.color == chess.WHITE:
                        piece_color = Colors.WHITE_PIECE
                    else:
                        piece_color = Colors.BLACK_PIECE

                    piece_symbol = get_piece_symbol(piece, use_unicode).strip()
                    line += f"{bg_color}{piece_color} {piece_symbol} {Colors.RESET}│"
                else:
                    line += f"{bg_color}   {Colors.RESET}│"
            else:
                piece_symbol = get_piece_symbol(piece, use_unicode).strip() if piece else ' '
                line += f" {piece_symbol} │"

        # Trailing rank label
        if use_colors:
            line += f" {Colors.COORD}{rank + 1}{Colors.RESET}"
        else:
            line += f" {rank + 1}"

        lines.append(line)

        # Add separator between ranks (except after rank 1)
        if rank > 0:
            lines.append("    ├───┼───┼───┼───┼───┼───┼───┼───┤")

    # Bottom border of board
    lines.append("    └───┴───┴───┴───┴───┴───┴───┴───┘")

    # Footer with file labels
    if use_colors:
        lines.append(f"{Colors.COORD}      a   b   c   d   e   f   g   h{Colors.RESET}")
    else:
        lines.append("      a   b   c   d   e   f   g   h")

    # Add captured pieces if provided
    if captured_pieces:
        if captured_pieces['white']:
            white_caps = ' '.join([get_piece_symbol(p, use_unicode).strip() for p in captured_pieces['white']])
            lines.append(f"\nCaptured by Black: {white_caps}")
        if captured_pieces['black']:
            black_caps = ' '.join([get_piece_symbol(p, use_unicode).strip() for p in captured_pieces['black']])
            lines.append(f"Captured by White: {black_caps}")

    return '\n'.join(lines)


def display_board_fancy(board, last_move=None, captured_pieces=None, use_unicode=True, use_colors=True, size='compact'):
    """
    Display a chess board with fancy Unicode pieces and colors.

    Args:
        board: chess.Board object
        last_move: chess.Move object to highlight
        captured_pieces: Dict with 'white' and 'black' lists of captured pieces
        use_unicode: Use Unicode pieces (vs ASCII)
        use_colors: Use ANSI colors (vs plain text)
        size: Display size - 'compact' (default, 8 lines) or 'large' (with borders, 17 lines)

    Returns:
        String representation of the board
    """
    if size == 'large':
        return display_board_large(board, last_move, captured_pieces, use_unicode, use_colors)

    # Original compact display
    lines = []

    # Determine which squares to highlight
    highlight_squares = set()
    if last_move:
        highlight_squares = {last_move.from_square, last_move.to_square}

    # Header with file labels
    if use_colors:
        lines.append(f"{Colors.COORD}    a  b  c  d  e  f  g  h{Colors.RESET}")
    else:
        lines.append("    a  b  c  d  e  f  g  h")

    # Board ranks (8 to 1)
    for rank in range(7, -1, -1):
        # Rank label
        if use_colors:
            rank_label = f"{Colors.COORD}{rank + 1}{Colors.RESET}"
        else:
            rank_label = f"{rank + 1}"

        line = f" {rank_label} "

        # Files (a to h)
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            # Choose background color
            if use_colors:
                if square in highlight_squares:
                    bg_color = Colors.HIGHLIGHT
                elif is_light_square(square):
                    bg_color = Colors.LIGHT_SQUARE
                else:
                    bg_color = Colors.DARK_SQUARE

                # Choose piece color
                if piece:
                    if piece.color == chess.WHITE:
                        piece_color = Colors.WHITE_PIECE
                    else:
                        piece_color = Colors.BLACK_PIECE
                else:
                    piece_color = ''

                piece_symbol = get_piece_symbol(piece, use_unicode)
                line += f"{bg_color}{piece_color}{piece_symbol}{Colors.RESET}"
            else:
                piece_symbol = get_piece_symbol(piece, use_unicode)
                line += piece_symbol

        # Trailing rank label
        if use_colors:
            line += f" {Colors.COORD}{rank + 1}{Colors.RESET}"
        else:
            line += f" {rank + 1}"

        lines.append(line)

    # Footer with file labels
    if use_colors:
        lines.append(f"{Colors.COORD}    a  b  c  d  e  f  g  h{Colors.RESET}")
    else:
        lines.append("    a  b  c  d  e  f  g  h")

    # Add captured pieces if provided
    if captured_pieces:
        if captured_pieces['white']:
            white_caps = ' '.join([get_piece_symbol(p, use_unicode).strip() for p in captured_pieces['white']])
            lines.append(f"\nCaptured by Black: {white_caps}")
        if captured_pieces['black']:
            black_caps = ' '.join([get_piece_symbol(p, use_unicode).strip() for p in captured_pieces['black']])
            lines.append(f"Captured by White: {black_caps}")

    return '\n'.join(lines)


def track_captured_pieces(board):
    """
    Calculate which pieces have been captured.

    Args:
        board: chess.Board object

    Returns:
        Dict with 'white' and 'black' lists of captured pieces
    """
    # Starting material
    starting_pieces = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
        chess.KING: 1,
    }

    # Count current pieces
    white_pieces = {pt: 0 for pt in chess.PIECE_TYPES}
    black_pieces = {pt: 0 for pt in chess.PIECE_TYPES}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_pieces[piece.piece_type] += 1
            else:
                black_pieces[piece.piece_type] += 1

    # Calculate captured pieces
    captured = {'white': [], 'black': []}

    for piece_type in chess.PIECE_TYPES:
        if piece_type == chess.KING:
            continue  # Kings can't be captured

        # White pieces captured by Black
        white_missing = starting_pieces[piece_type] - white_pieces[piece_type]
        for _ in range(white_missing):
            captured['white'].append(chess.Piece(piece_type, chess.WHITE))

        # Black pieces captured by White
        black_missing = starting_pieces[piece_type] - black_pieces[piece_type]
        for _ in range(black_missing):
            captured['black'].append(chess.Piece(piece_type, chess.BLACK))

    return captured


if __name__ == "__main__":
    # Test the display
    print("Testing enhanced board display...\n")

    # Starting position - compact
    board = chess.Board()
    print("="*50)
    print("COMPACT MODE (default)")
    print("="*50)
    print("\nStarting position:")
    print(display_board_fancy(board, size='compact'))

    # After a few moves - compact
    print("\n\nAfter e4 e5 Nf3 Nc6 Bb5:")
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    last = board.push_san("Bb5")

    captured = track_captured_pieces(board)
    print(display_board_fancy(board, last_move=last, captured_pieces=captured, size='compact'))

    # Now test large mode
    print("\n\n" + "="*50)
    print("LARGE MODE (with box-drawing borders)")
    print("="*50)

    # Starting position - large
    board = chess.Board()
    print("\nStarting position:")
    print(display_board_fancy(board, size='large'))

    # After a few moves - large
    print("\n\nAfter e4 e5 Nf3 Nc6 Bb5:")
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    last = board.push_san("Bb5")

    captured = track_captured_pieces(board)
    print(display_board_fancy(board, last_move=last, captured_pieces=captured, size='large'))

    # Position with captures - large
    print("\n\nPosition with captures:")
    board = chess.Board()
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6"]
    for move_san in moves:
        last = board.push_san(move_san)

    captured = track_captured_pieces(board)
    print(display_board_fancy(board, last_move=last, captured_pieces=captured, size='large'))

    print("\n✅ Display test complete!")
