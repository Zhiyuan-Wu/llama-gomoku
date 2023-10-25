import logging
import math

import numpy as np
import time
import torch

EPS = 1e-8
CUDA_AVAILABLE = torch.cuda.is_available()

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.gpu_accumulate_time = 0
        self.device = next(nnet.parameters()).device

    def getActionProb(self, canonicalBoard, temp=1, analyze=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        winrate = [self.Qsa[(s, a)] if (s, a) in self.Qsa else -1 for a in range(self.game.getActionSize())]
        pv = []
        for a in range(self.game.getActionSize()):
            _pv = []
            if counts[a] > 0:
                _current = canonicalBoard
                _a = a
                for _ in range(20):
                    _current, _ = self.game.getNextState(_current, 1, _a)
                    _current = self.game.getCanonicalForm(_current, -1)
                    _s = self.game.stringRepresentation(_current)
                    _counts = [self.Nsa[(_s, a)] if (_s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
                    if np.any(np.array(_counts)>0):
                        _a = int(np.argmax(_counts))
                        _pv.append(_a)
                    else:
                        break
            pv.append(_pv)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        if analyze:
            return probs, winrate, pv
        else:
            return probs

    def search(self, canonicalBoard, root_flag=True):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s] * self.args.endGameRewardWeight

        if s not in self.Ps:
            # leaf node
            _record_time = time.time()
            _b = torch.FloatTensor(canonicalBoard.astype(np.float64))
            if CUDA_AVAILABLE: _b = _b.contiguous().cuda(self.device)
            _b = _b.view(-1, self.game.n, self.game.n)
            self.nnet.eval()
            with torch.no_grad():
                _pi, _v = self.nnet(_b)
                _pi = torch.exp(_pi).data.cpu().numpy().squeeze()
                _v = torch.softmax(_v, -1).data.cpu().numpy().squeeze()
            self.Ps[s], v = _pi, _v
            v = v[0]-v[1]+1e-4*v[2]
            self.gpu_accumulate_time += time.time() - _record_time
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        _dirichlet_noise = np.random.dirichlet(self.args.dirichlet_alpha * np.ones((self.game.getActionSize())))
        for a in range(self.game.getActionSize()):
            if valids[a]:
                _Psa = self.Ps[s][a]
                if root_flag:
                    _Psa = (1-self.args.dirichlet_weight) * _Psa + self.args.dirichlet_weight * _dirichlet_noise[a]
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * _Psa * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * _Psa * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, False)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

class batch_MCTS():
    """
    This class handles the MCTS tree. allow batch evluation
    """

    def __init__(self, game, args, shared_Ps, shared_Es, shared_Vs, query_buffer, identifier):
        self.game = game
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = shared_Ps  # stores initial policy (returned by neural net), shared between workers.

        self.Es = shared_Es  # stores game.getGameEnded ended for board s, shared between workers.
        self.Vs = shared_Vs  # stores game.getValidMoves for board s, shared between workers.
        self.qb = query_buffer # stores board to be evaluated
        self.identifier = identifier # worker id

        self.board = game.getInitBoard()
        self.player = 1
        self.episodeStep = 0
        self.game_record = []
        self.search_path = []
        self.current_state = game.getCanonicalForm(self.board, self.player)
        self.current_value = None

        self.search_count = 1
        self.total_search_depth = 1

    def reset(self):
        ''' Reset the tree state, ready for new search (under same policy). Note that this will not reset the shared content (Ps, Es, Vs).
        '''
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()

        self.board = self.game.getInitBoard()
        self.player = 1
        self.episodeStep = 0
        self.game_record = []
        self.search_path = []
        self.current_state = self.game.getCanonicalForm(self.board, self.player)
        self.current_value = None

        self.search_count = 1
        self.total_search_depth = 1
    
    def cpuct(self, s, root_flag):
        valids = self.Vs[s]
        if s in self.Qsa:
            qsa = self.Qsa[s]
            nsa = self.Nsa[s]
        else:
            qsa = np.zeros((self.game.getActionSize()), dtype=np.float16)
            nsa = np.zeros((self.game.getActionSize()), dtype=np.uint16)

        _Ps = self.Ps[s]
        if root_flag:
            # no need to normalize again
            _Ps = (1-self.args.dirichlet_weight) * _Ps + self.args.dirichlet_weight * np.random.dirichlet(self.args.dirichlet_alpha * np.ones((self.game.getActionSize())))
        u = qsa + self.args.cpuct * _Ps * (np.sqrt(self.Ns[s]) + EPS) / (1+nsa)
        a = np.argmax((u + 999999)*valids)
        return a
    
    def extend(self):
        ''' excute a forward search, stop at a leaf node (non-evaluated node or terminal-node).
        '''
        self.current_state = self.game.getCanonicalForm(self.board, self.player)
        root_flag = True
        while 1:
            s = self.game.stringRepresentation(self.current_state)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(self.current_state, 1)
            if self.Es[s] != 0:
                # terminal node
                self.current_value = -self.Es[s] * self.args.endGameRewardWeight
                break
            
            if s not in self.Ps:
                self.qb.append([self.identifier, self.current_state, s])
                self.current_value = None
                self.Ns[s] = 0
                break

            if s not in self.Ns:
                self.Ns[s] = 0

            # pick the action with the highest upper confidence bound
            if s not in self.Vs:
                self.Vs[s] = self.game.getValidMoves(self.current_state, 1)
            best_act = self.cpuct(s, root_flag)

            next_s, next_player = self.game.getNextState(self.current_state, 1, best_act)
            self.current_state = self.game.getCanonicalForm(next_s, next_player)

            self.search_path.append((s, best_act))
            root_flag = False

    def _set_result(self):
        ''' provides a fake function that set all queries from extend() with dummy answers. only for debug purpose.
        '''
        if len(self.qb)>0:
            query_index, query_content, query_state_string = zip(*self.qb)
            pi = np.ones((self.game.getActionSize()))
            v = 0.001
            for j,s in enumerate(query_state_string):
                # Set Ps and Vs
                self.Ps[s] = pi
                if s not in self.Vs:
                    valids = self.game.getValidMoves(query_content[j], 1)
                    self.Vs[s] = valids
                else:
                    valids = self.Vs[s]
                self.Ps[s] = self.Ps[s] * valids
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:  
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])
                
                # Set values
                self.current_value = -v
            self.qb.clear()
    
    def backprop(self):
        ''' after self.current_value be set (either by extend() for terminal-node, or outside controller for non-evaluated node), update tree statics
        '''
        self.search_count += 1
        self.total_search_depth += len(self.search_path)
        for s,a in self.search_path[::-1]:
            if s in self.Qsa:
                _qsa = self.Qsa[s][a]
                _nsa = self.Nsa[s][a]
                self.Qsa[s][a] = (_nsa * _qsa + self.current_value) / (_nsa + 1)
                self.Nsa[s][a] = _nsa + 1
            else:
                self.Qsa[s] = np.zeros((self.game.getActionSize(),), dtype=np.float16)
                self.Qsa[s][a] = self.current_value
                # self.Qsa[s] = np.ones((self.game.getActionSize(),)) * self.current_value
                self.Nsa[s] = np.zeros((self.game.getActionSize(),), dtype=np.uint16)
                self.Nsa[s][a] = 1

            self.Ns[s] += 1
            self.current_value = -self.current_value
        self.search_path.clear()
