from gobang.GobangNNet import GobangNNet, GobangNNetArgs
from gobang.GobangGame import GobangGame
from utils import *
from MCTS import MCTS
import numpy as np
import json
import tqdm
import torch
import torch.multiprocessing as mp

class Arena():
    def __init__(self, ckpt_path, gpuid):
        self.gpuid = gpuid
        self.load_model(ckpt_path)

    def load_model(self, ckpt_path):
        self.model_args = GobangNNetArgs()
        self.game = GobangGame()
        self.model = GobangNNet(self.game, self.model_args)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda(self.gpuid).eval()

    def selfplay(self):
        mcts = MCTS(self.game, self.model, self.model_args)
        board = np.zeros((self.game.n, self.game.n), dtype=np.int8)
        cur_player = 1
        move_count = 0
        record = {}
        move_history = []
        while 1:
            _player = mcts
            canonicalBoard = self.game.getCanonicalForm(board, cur_player)
            pi, winrate, pv = _player.getActionProb(canonicalBoard, self.model_args.tempereature, True)
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            move_history.append(action)
            move_count += 1
            record[f"move{move_count}"] = {"canonicalBoard": canonicalBoard.tolist(),
                                           "pi": [int(x*10000) for x in pi],
                                           "action": action,
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
    arena = Arena("model/checkpoint_n15b6c128i3000d230912.pth", rank%gpu_num)
    progress_bar = tqdm.tqdm(desc=f"Self-Play #{rank}", total=sample_num, position=rank)
    progress_bar.set_lock(lock)
    with open(f"data/selfplay_data_{rank}.json", 'a') as fout:
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

