import psutil
import random
import numpy as np

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    """Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

def action2str(action, n=15):
    return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[action%n], action//n+1)

action2strl = lambda y: [action2str(x) for x in y]

def find_available_port(port):
    while 1:
        port = random.randint(20000, 60000)
        with psutil.net_connections() as connections:
            # Check if the port is in use by any connection
            for conn in connections:
                if conn.laddr.port == port:
                # Port is not available
                    continue
            # Port is available
            break
    return port

def check_pattern(board, x, y, pattern, direction):
    ''' Determine if the x,y position of board staisfy given pattern on given direction
    '''
    target_index = pattern.index(2)
    left_range = target_index
    right_range = len(pattern) - left_range - 1
    for i in range(1, left_range + 1):
        tx = x - i * direction[0]
        ty = y - i * direction[1]
        if (not 0<=tx<15) or (not 0<=ty<15):
            return False
        if not board[tx, ty] == pattern[target_index - i]:
            return False
    for i in range(1, right_range + 1):
        tx = x + i * direction[0]
        ty = y + i * direction[1]
        if (not 0<=tx<15) or (not 0<=ty<15):
            return False
        if not board[tx, ty] == pattern[target_index + i]:
            return False
    return True

def compute_interest_points(board, color):
    ''' This function determines the interest_points of a board for a player.
        return value indicates:
        - 3: open-three point
        - 4: open-four point
        - 5: win point
        - 6: double-open-three point
        - 7: open-four-three point
        - 8: double-open-four point
    '''
    if color==-1:
        board = - board
    interest_board = np.zeros([15,15,4])
    patterns_o3 = [[0, 2, 1, 1, 0], [0, 1, 2, 1, 0], [0, 1, 1, 2, 0],
                    [0, 2, 1, 0, 1, 0], [0, 1, 2, 0, 1, 0], [0, 1, 1, 0, 2, 0],
                      [0, 2, 0, 1, 1, 0], [0, 1, 0, 2, 1, 0], [0, 1, 0, 1, 2, 0], ]
    for i in range(15):
        for j in range(15):
            if board[i,j]!=0:
                continue
            for k,d in enumerate([[1,0], [0,1], [1,1], [1,-1]]):
                for p in patterns_o3:
                    if check_pattern(board, i, j, p, d):
                        interest_board[i, j, k] = 3
                        break
    patterns_o4 = [[0, 2, 1, 1, 1], [0, 1, 2, 1, 1], [0, 1, 1, 2, 1], [0, 1, 1, 1, 2], 
                   [2, 0, 1, 1, 1], [1, 0, 2, 1, 1], [1, 0, 1, 2, 1], [1, 0, 1, 1, 2], 
                   [2, 1, 0, 1, 1], [1, 2, 0, 1, 1], [1, 1, 0, 2, 1], [1, 1, 0, 1, 2], 
                   [2, 1, 1, 0, 1], [1, 2, 1, 0, 1], [1, 1, 2, 0, 1], [1, 1, 1, 0, 2], 
                   [2, 1, 1, 1, 0], [1, 2, 1, 1, 0], [1, 1, 2, 1, 0], [1, 1, 1, 2, 0],]
    for i in range(15):
        for j in range(15):
            if board[i,j]!=0:
                continue
            for k,d in enumerate([[1,0], [0,1], [1,1], [1,-1]]):
                for p in patterns_o4:
                    if check_pattern(board, i, j, p, d):
                        interest_board[i, j, k] = 4
                        break

    patterns_o5 = [[2, 1, 1, 1, 1], [1, 2, 1, 1, 1], [1, 1, 2, 1, 1], [1, 1, 1, 2, 1], [1, 1, 1, 1, 2],]
    for i in range(15):
        for j in range(15):
            if board[i,j]!=0:
                continue
            for k,d in enumerate([[1,0], [0,1], [1,1], [1,-1]]):
                for p in patterns_o5:
                    if check_pattern(board, i, j, p, d):
                        interest_board[i, j, k] = 5
                        break

    result = np.zeros([15,15])
    index2pattern = {3: "Open-Three", 4: "Open-Four", 5: "Win-Five", 6: "Double-Open-Three", 7: "Open-Four-Three", 8: "Double-Open-Four", }
    readable = {'Open-Three': [], 'Open-Four': [], 'Win-Five': [], 'Double-Open-Three': [], 'Open-Four-Three': [], 'Double-Open-Four': [], }
    for i in range(15):
        for j in range(15):
            num_of_3 = np.sum(interest_board[i, j] == 3)
            num_of_4 = np.sum(interest_board[i, j] == 4)
            num_of_5 = np.sum(interest_board[i, j] == 5)
            if num_of_3 > 1:
                result[i,j] = 6
            elif num_of_3 > 0:
                result[i,j] = 3
            if num_of_4 > 1:
                result[i,j] = 8
            elif num_of_4 > 0:
                if num_of_3 > 0:
                    result[i,j] = 7
                else:
                    result[i,j] = 4
            if num_of_5 > 0:
                result[i,j] = 5

            if result[i,j]>0:
                readable[index2pattern[result[i,j]]].append(action2str(i*15+j))

    return result, readable