import torch
import torch.nn as nn
import pdb

def get_criterion(config, vocab):
    
    if config.model.name == 'las':
        criterion = Attention_Loss(config, vocab)
    elif config.model.name in ['conf', 'conf_a']:
        if config.decoder.method=='att_only':
            criterion = Attention_Loss(config, vocab)
        elif config.decoder.method=='ctc_only':
            criterion = CTC_Loss(config, vocab)
        else:
            criterion = CTC_Attention_Loss(config, vocab)
    return criterion

class CTC_Attention_Loss(nn.Module):
    '''
    Inputs : 
        outputs : tuple ( tensor (BxSxE), tensor (BxLxE) )
        targets : tensor (BxS)
    '''
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.ctc   = nn.CTCLoss(blank=vocab.unk_id, reduction='mean', zero_infinity=True)
        self.att   = LabelSmoothingLoss(len(vocab), ignore_index=vocab.pad_id,
                                        smoothing=config.model.label_smoothing)
        
    def forward(self, outputs, targets, target_lengths):
        a = float(self.config.model.alpha)
        targets = targets
        att_out = outputs[0].contiguous().view(-1,outputs[0].shape[-1]) 
        ctc_out = outputs[1].contiguous().permute(1,0,2) # (B,L,E)->(L,B,E)
        att_loss = self.att(att_out, targets.contiguous().view(-1))
        if ctc_out.shape[1] < target_lenghts.max():
            pdb.set_trace()
        ctc_loss = self.ctc(ctc_out, targets, # ctc_out.size(0), targets.size(1),
                            (torch.ones(ctc_out.shape[1])*ctc_out.shape[0]).to(torch.int),
                            target_lengths) 
        return a*att_loss + (1-a)*ctc_loss
        
        
class CTC_Loss(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.ctc   = nn.CTCLoss(blank=vocab.unk_id, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, targets, target_lengths):
        ctc_out = outputs.contiguous().permute(1,0,2) # (B,L,E)->(L,B,E)
        pdb.set_trace()
        ctc_loss = self.ctc(ctc_out, targets, # ctc_out.size(0), targets.size(1),
                            (torch.ones(ctc_out.shape[1])*ctc_out.shape[0]).to(torch.int),
                            target_lengths) 
        return ctc_loss
        
        
class Attention_Loss(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.att   = LabelSmoothingLoss(len(vocab), ignore_index=vocab.pad_id,
                                        smoothing=config.model.label_smoothing)
        
    def forward(self, outputs, targets, *args, **kwargs):
        out = outputs.contiguous().view(-1,outputs.shape[-1])
        loss = self.att(out, targets.contiguous().view(-1))
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        logit = logit.view(-1, logit.shape[-1])
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit).cuda()
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0
        # return torch.sum(-label_smoothed * logit)
        return torch.mean(torch.sum(-label_smoothed * logit, axis=-1))