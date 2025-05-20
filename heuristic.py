import torch 
import numpy as np

def heuristic(matrix):
    """
    Returns:
        valid_moves (list): directions that are safe
        moves (dict): boolean flags for each direction
        action_mask (torch.BoolTensor): [up, down, left, right] mask
    """
    height, width, _ = matrix.shape
    moves = {"up": True, "down": True, "left": True, "right": True}

    # Locate head
    head_pos = np.argwhere(matrix[:, :, 1] == 5)
    if len(head_pos) == 0:
        return [], moves, torch.tensor([True, True, True, True], dtype=torch.bool)

    head_y, head_x = head_pos[0]

    # Do not walk into walls
    if head_y == 0:
        moves["up"] = False
    if head_y == height - 1:
        moves["down"] = False
    if head_x == 0:
        moves["left"] = False
    if head_x == width - 1:
        moves["right"] = False

    # Neck detection (look for adjacent segment with value 1 in channel 1)
    for dy, dx, direction in [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]:
        ny, nx = head_y + dy, head_x + dx
        if 0 <= ny < height and 0 <= nx < width and matrix[ny, nx, 1] == 1:
            moves[direction] = False
            #break   Only block the first neck segment found

    valid_moves = [move for move, ok in moves.items() if ok]

    action_mask = torch.tensor([
        not moves["up"],
        not moves["down"],
        not moves["left"],
        not moves["right"]
    ], dtype=torch.bool)

    if not valid_moves:
        print("No valid moves available")
        valid_moves = ["up", "down", "left", "right"]

    return valid_moves, moves, action_mask
