from gobang.GobangNNet import GobangNNet, GobangNNetArgs
from gobang.GobangGame import GobangGame
from utils import *
from MCTS import MCTS
import numpy as np
import json
import tqdm
from pathlib import Path
import random
import torch
import torch.multiprocessing as mp

class Arena():
    def __init__(self, ckpt_path_dir, gpuid):
        self.gpuid = gpuid
        self.ckpt_path_dir = ckpt_path_dir
        self.prepare_model()

    def prepare_model(self):
        self.checkpoints = sorted(Path(self.ckpt_path_dir).glob("*.pth"))
        self.model_args = GobangNNetArgs()
        self.game = GobangGame()
        self.set_init_boards()
        self.model1 = GobangNNet(self.game, self.model_args).cuda(self.gpuid)
        self.model2 = GobangNNet(self.game, self.model_args).cuda(self.gpuid)
    
    def set_init_boards(self):
        self.init_board = [np.zeros((self.game.n, self.game.n), dtype=np.int8)]

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = 1
        _b[6, 7] = -1
        self.init_board.append(_b)
        
        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = 1
        _b[6, 8] = -1
        self.init_board.append(_b)
        
        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[6, 8] = 1
        _b[10, 6] = -1
        self.init_board.append(_b)

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[6, 7] = 1
        _b[5, 9] = -1
        self.init_board.append(_b)

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[9, 6] = 1
        _b[10, 4] = -1
        self.init_board.append(_b)

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[7, 6] = 1
        _b[8, 10] = -1
        self.init_board.append(_b)

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[9, 6] = 1
        _b[8, 4] = -1
        self.init_board.append(_b)

        _b = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        _b[7, 7] = -1
        _b[5, 8] = 1
        _b[8, 4] = -1
        self.init_board.append(_b)

    def selfplay(self):
        player1 = random.choice(self.checkpoints)
        state_dict = torch.load(player1, map_location=f'cuda:{self.gpuid}')
        consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        self.model1.load_state_dict(state_dict)
        self.model1.eval()
        mcts1 = MCTS(self.game, self.model1, self.model_args)

        player2 = random.choice(self.checkpoints)
        state_dict = torch.load(player2, map_location=f'cuda:{self.gpuid}')
        consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        self.model2.load_state_dict(state_dict)
        self.model2.eval()
        mcts2 = MCTS(self.game, self.model2, self.model_args)

        board = np.array(random.choice(self.init_board))
        cur_player = 1
        move_count = 0
        record = {}
        move_history = []
        while 1:
            _player = {1: mcts1, -1:mcts2}[cur_player]
            canonicalBoard = self.game.getCanonicalForm(board, cur_player)
            pi, counts, winrate, pv = _player.getActionProb(canonicalBoard, self.model_args.tempereature, True)
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            move_history.append(action)
            move_count += 1
            record[f"move{move_count}"] = {"canonicalBoard": canonicalBoard.tolist(),
                                           "action": action,
                                           "counts": [int(x*10000) for x in counts],
                                           "winrate": [int(x*10000) for x in winrate],
                                           "pv": pv,
                                           }
            r = self.game.getGameEnded(board, 0)
            if r!=0:
                record["move_count"] = move_count
                record["result"] = r
                record["move_history"] = move_history
                break
        return record

def worker(sample_num, rank, gpu_num, lock):
    arena = Arena("/home/zhiyuan/others/alpha-zero-general/result0816", rank%gpu_num)
    progress_bar = tqdm.tqdm(desc=f"Self-Play #{rank}", total=sample_num, position=rank)
    progress_bar.set_lock(lock)
    with open(f"data/selfplay/selfplay_data_{rank}.json", 'w') as fout:
        for _ in range(sample_num):
            fout.write(json.dumps(arena.selfplay()) + '\n')
            progress_bar.update(1)

if __name__=="__main__":
    mp.set_start_method('spawn')
    lock = mp.RLock()
    for i in range(12):
        p = mp.Process(target=worker, args=(1000, i, 4, lock))
        p.start()
    for _ in range(12):
        p.join()

