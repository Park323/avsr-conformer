import torch
import torch.nn.functional as F
import numpy as np

class HybridInference(object):
    def __init__(self, config, vocab):
        self.config = config
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.unk_id = vocab.unk_id
        self.pad_id = vocab.pad_id
        self.ctc_att_rate = config.decoder.ctc_att_rate
        self.beam_width = config.decoder.beam_width
        self.vocab_size = config.decoder.vocab_size
        
    def __call__(self, features):
        return self.onePassBeamSearch(features)
        
    def onePassBeamSearch(self, 
                      features,
                      *args, **kwargs):
        # omega_0
        queue = [torch.tensor([self.sos_id])]
        scores = [0]
        beam_width = self.beam_width
        
        complete = []
        complete_scores= []
        
        # get CTC prediction
        ctcProb = F.softmax(self.ctcLinear(features), dim=-1)
        
        Y_n = {t:{(self.sos_id,) : torch.tensor(0)} 
                  for t in range(features.size(1))}
        Y_b = {t:{(self.sos_id,) : self.pi_sos(t, ctcProb[0])}
                  for t in range(features.size(1))}
        
        for l in range(1, min(features.size(1), self.config.model.max_len)):
            n_queue = []
            n_scores = []
            
            while queue:
                # _t = time()
                g = queue.pop()
        
                # get Attention prediction
                labels = F.one_hot(g, self.vocab_size).unsqueeze(0).to(features.device)
                labels = self.target_embedding(labels.to(torch.float32))
                attProb = self.decoder(labels, features, train=False)
                attScore = F.log_softmax(self.ceLinear(attProb),dim=-1)
                # print(f'Attention decoding took {time()-_t:.4f}sec...')
                
                candidate_idxs = attScore.topk(k=int(self.vocab_size*0.05), dim=-1).indices[0,-1,:].tolist()
                candidate_idxs = [self.eos_id, *candidate_idxs]
                # _t = time()
                for c in candidate_idxs:
                    # Loop Except sos token
                    if c==self.sos_id:
                        continue
                    elif c==self.unk_id:
                        continue
                        
                    h = torch.cat([g,torch.tensor([c])])
                    
                    ctc = self.get_ctc_score(tuple(h.tolist()), ctcProb.squeeze(0), 
                                             Y_n, Y_b)
                    att = attScore[0,-1,c]
                    score = self.ctc_att_rate * ctc + (1-self.ctc_att_rate) * att
                    score = score.cpu().item()
                    
                    if c==self.eos_id:
                        complete_scores.append(score)
                        complete.append(h)
                    else:
                        n_scores.append(score)
                        n_queue.append(h)
                        if len(n_queue) > beam_width:
                            min_idx = np.argmin(n_scores)
                            n_scores.pop(min_idx)
                            n_queue.pop(min_idx)
                # print(f"CTC decoding took {time()-_t:.4f}sec...")
                
            if self.EndDetect(l+1, complete, complete_scores):
                break

            queue = n_queue
            scores = n_scores
        
        # free    
        del Y_n
        del Y_b
        
        best_guess = complete[np.argmax(complete_scores)]
        pad        = torch.ones(150-best_guess.size(0)).to(best_guess.device) * self.pad_id
        best_guess = torch.cat([best_guess, pad.int()], dim=0)
        best_guess = F.one_hot(best_guess, self.vocab_size)
        best_guess = best_guess.unsqueeze(0)
        
        return best_guess

    def pi_sos(self, t, prob):
        y = 1
        for i in range(t):
            y *= prob[i, self.unk_id]
        return y 
            
    def EndDetect(self, l, complete_list, complete_score, D=-2, M=3):
        if complete_score:
            global_max = np.max(complete_score)
            for m in range(min(M, l)):
                idxs = []
                seq_list = []
                score_list = []
                for i, a in enumerate(complete_list):
                    if len(a) == l - m:
                        idxs.append(i)
                        score_list.append(complete_score[i])
                max_score = np.max(score_list)
                if max_score - global_max >= D:
                    return False
        return True
                
    def get_ctc_score(self, h, X, 
                      Y_n=None, Y_b=None):
        ## h : (L, 1)
        ## X : (T, E)
        T = X.size(0)
        L = len(h)
        g = h[:-1]
        c = h[-1]
        if c == self.eos_id:
            return np.log((Y_n[T-1][g] + Y_b[T-1][g]).cpu().item() + EPSILON)
        elif T == 1:
            return np.log((Y_n[T][h] + Y_b[T][h]).cpu().item())
        else:
            Y_n[L-1][h] = X[0, c] if g == (self.sos_id,) else 0
            Y_b[L-1][h] = 0
            # psi : Probability for each seq_length
            psi = Y_n[L-1][h]
            for t in range(L-1, T):
                # phi : ctc probability for 'g' before 'c'
                phi = Y_b[t-1][g] + (0 if g[-1]==c else Y_n[t-1][g])
                Y_n[t][h] = (Y_n[t-1].get(h, 0) + phi) * X[t, c]
                Y_b[t][h] = (Y_b[t-1].get(h, 0) + Y_n[t-1].get(h, 0)) * X[t, self.unk_id]
                psi = psi + phi * X[t, c]
            return np.log(psi.cpu().item() + EPSILON)