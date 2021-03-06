import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.conformer.encoder import ConformerEncoder

'''
Video : 30 fps
Audio : 16000 Hz
'''

EPSILON = 1e-100


class BaseConformer(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.unk_id = vocab.unk_id
        self.pad_id = vocab.pad_id
        self.vocab_size = len(vocab)
        self.alpha = config.model.alpha
        
        self.target_embedding = nn.Linear(self.vocab_size, config.decoder.d_model)
        self.decoder= TransformerDecoder(config)
        self.ceLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        self.ctcLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                targets=None, target_lengths=None, 
                *args, **kwargs):
        features = self.encode(video_inputs, video_input_lengths,
                               audio_inputs, audio_input_lengths)
        if targets is None:
            outputs, output_lengths = self.greedySearch(features)
            return outputs, output_lengths
        else:
            targets = F.one_hot(targets, num_classes = self.vocab_size)
            targets = self.target_embedding(targets.to(torch.float32))
            ctc_out = F.log_softmax(self.ctcLinear(features), dim=-1)
            if self.config.decoder.method == 'ctc_only':
                return ctc_out
            att_out = F.log_softmax(self.ceLinear(
                self.decoder(targets, features, pad_id=self.pad_id)
                ), dim=-1)
            if self.config.decoder.method == 'att_only':
                return att_out
            return (att_out, ctc_out)
            
    def encode(self,
               video_inputs, video_input_lengths,
               audio_inputs, audio_input_lengths, 
               *args, **kwargs):
        pass

    def greedySearch(self, features, *args, **kwargs):
        if self.config.decoder.method=='att_only':
            return self.greedyAttentionSearch(features)
        elif self.config.decoder.method=='ctc_only':
            return self.greedyCTCSearch(features)
        else:
            return self.greedyHybridSearch(features)

    def greedyCTCSearch(self, features, *args, **kwargs):
        max_len = self.config.model.max_len
        
        outputs = F.log_softmax(self.ctcLinear(features), dim=-1)
        preds_with_blank = torch.argmax(outputs, dim=-1)
        preds = torch.full((features.size(0), max_len+1), self.unk_id).to(features.device).to(int)
        pred_lengths = torch.zeros(outputs.size(0), dtype=int, device=preds.device)
        
        for i, y_hat in enumerate(preds_with_blank):
            for letter in y_hat:
                if letter == self.eos_id:
                    break
                # unk_id
                if preds[i,pred_lengths[i]] == self.unk_id:
                    pass
                # repeatetion
                elif pred_lengths[i] > 0 and preds[i,pred_lengths[i]-1] == letter:
                    continue
                
                preds[i,pred_lengths[i]] = letter
                if letter == self.unk_id:
                    pass
                else:
                    pred_lengths[i] += 1
                
        preds = F.one_hot(preds[:,1:], self.vocab_size).to(float)
        preds = F.log_softmax(preds, dim=-1)
        
        return preds, pred_lengths

    def greedyAttentionSearch(self, features, *args, **kwargs):
        max_len = self.config.model.max_len
        preds = torch.full((features.size(0), max_len+1), 0).to(features.device).to(int)
        pred_lengths = torch.full((features.size(0),), 0, dtype=int, device=features.device)
        active_batch = torch.full((features.size(0),), True, dtype=bool, device=features.device)
                
        loop_idx = 0
        while loop_idx <= max_len:
            # pdb.set_trace()
            # End the loop
            if active_batch.sum()==0:
                break 
            # Fill sos_id
            elif loop_idx == 0:
                preds[:,loop_idx] = 1
            # if unfinished batch exists
            else:
                targets = F.one_hot(preds[active_batch,:loop_idx], num_classes = self.vocab_size)
                targets = self.target_embedding(targets.to(torch.float32))
                outputs = self.decoder(targets, features[active_batch], train=False, pad_id=0)
                outputs = self.ceLinear(outputs)
                topk_ids = outputs.topk(k=1, dim=-1).indices[:,-1,0]
                preds[active_batch, loop_idx] = topk_ids
                
                pred_lengths[active_batch] += 1
                active_batch[preds[:, loop_idx] == 2] = False
                
            loop_idx += 1
            
        preds = F.one_hot(preds[:,1:], self.vocab_size).to(float)
        preds = F.log_softmax(preds, dim=-1)
        
        return preds, pred_lengths
        
    def greedyHybridSearch(self, features, K=5, *args, **kwargs):
        max_len = self.config.model.max_len
        preds = torch.full((features.size(0), max_len+1), 0, dtype=int, device=features.device)
        pred_lengths = torch.full((features.size(0),), 0, dtype=int, device=features.device)
        active_batch = torch.full((features.size(0),), True, dtype=bool, device=features.device)
        ctc_outputs = F.softmax(self.ctcLinear(features), dim=-1)
        
        Y_n = [{t:{(1,) : torch.tensor(0)} for t in range(features.size(1))} 
               for i in range(features.size(0))]
        Y_b = [{t:{(1,) : self.pi_sos(t, ctc_outputs[i])} for t in range(features.size(1))}
               for i in range(features.size(0))]
                  
        loop_idx = 0
        while loop_idx <= max_len and loop_idx < ctc_outputs.size(1):
            # End the loop
            if active_batch.sum()==0:
                break 
            # Fill sos_id
            elif loop_idx == 0:
                preds[:,loop_idx] = 1
            # if unfinished batch exists
            else:
                ### attention
                targets = F.one_hot(preds[active_batch,:loop_idx], num_classes = self.vocab_size)
                targets = self.target_embedding(targets.to(torch.float32))
                outputs = self.decoder(targets, features[active_batch], train=False, pad_id=0)
                outputs = self.ceLinear(outputs)
                att_scores = F.log_softmax(outputs[:,-1], dim=-1)
                
                ### CTC
                ctc_scores = torch.zeros_like(att_scores)
                # candidate_idxs = range(self.vocab_size) # Full Search
                candidate_idxs = att_scores.topk(k = K, dim = -1).indices[:,:] # Partial Search
                for i, g in enumerate(preds[active_batch, :loop_idx]):
                    for c in candidate_idxs[i]:
                        h = torch.cat([g,torch.tensor([c], device=g.device)])
                        ctc_scores[i, c] = self.get_ctc_score(tuple(h.tolist()), ctc_outputs[i], Y_n[i], Y_b[i])
                
                ### Integrate scores
                scores = self.alpha * att_scores + (1-self.alpha) * ctc_scores
                topk_ids = scores.topk(k=1, dim=-1).indices[:,0]
                
                preds[active_batch, loop_idx] = topk_ids
                
                pred_lengths[active_batch] += 1
                active_batch[preds[:, loop_idx] == 2] = False
                
            loop_idx += 1
            
        preds = F.one_hot(preds[:,1:], self.vocab_size).to(float)
        preds = F.log_softmax(preds, dim=-1)
        
        return preds, pred_lengths
        
    def get_ctc_score(self, h, X, 
                      Y_n=None, Y_b=None):
        ## h : (L, 1)
        ## X : (T, E)
        T = X.size(0)
        L = len(h)
        g = h[:-1]
        c = h[-1]
        if c == 2:
            try: Y_b[T-1][g]
            except: self.get_ctc_score(g, X, Y_n, Y_b)
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
                try: Y_b[t-1][g]
                except: self.get_ctc_score(g, X, Y_n, Y_b)
                finally:
                    phi = Y_b[t-1][g] + (0 if g[-1]==c else Y_n[t-1][g])
                Y_n[t][h] = (Y_n[t-1].get(h, 0) + phi) * X[t, c]
                Y_b[t][h] = (Y_b[t-1].get(h, 0) + Y_n[t-1].get(h, 0)) * X[t, self.unk_id]
                psi = psi + phi * X[t, c]
            return np.log(psi.cpu().item() + EPSILON)

    def pi_sos(self, t, prob):
        y = 1
        for i in range(t):
            y *= prob[i, self.vocab.unk_id]
        return y
    

class AudioVisualConformer(BaseConformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        self.config = config
        self.fusion = FusionModule(config)
        self.visual = VisualFeatureExtractor(config)
        self.audio  = AudioFeatureExtractor(config)
        
    def encode(self, 
               video_inputs, video_input_lengths,
               audio_inputs, audio_input_lengths,
               *args, **kwargs):
        video_inputs, audio_inputs = self.match_seq(video_inputs, audio_inputs)
        audioFeatures  = self.audio(audio_inputs)
        visualFeatures = self.visual(video_inputs,
                                     video_input_lengths)
        outputs = self.fusion(visualFeatures, audioFeatures)
        return outputs
    
    def match_seq(self, video_inputs, audio_inputs):
        if 'lip' in self.config.train.transcripts_path_train:
            extend_length = int(video_inputs.size(1) * 30 / 29)
            extend_matrix = torch.zeros((video_inputs.size(0), extend_length, video_inputs.size(2)), device=video_inputs.device)
            for i in range(video_inputs.size(1) // 29):
                extend_matrix[:,i*30] = video_inputs[:,i*29] 
                extend_matrix[:,i*30+1:(i+1)*30] = video_inputs[:,i*29:(i+1)*29]
            _video_inputs = extend_matrix
        else:
            _video_inputs = video_inputs
        vid_len = _video_inputs.size(2) if self.config.video.use_raw_vid=='on' else _video_inputs.size(1)
        aud_seq_len = vid_len * 480
        if aud_seq_len <= audio_inputs.size(2):
            audio_outputs = audio_inputs[:, :, :aud_seq_len]
        else:
            pad = torch.zeros([*audio_inputs.shape[:2], aud_seq_len-audio_inputs.size(2)]).to(audio_inputs.device)
            audio_outputs = torch.cat([audio_inputs, pad], dim=2)
        return _video_inputs, audio_outputs


class AudioConformer(BaseConformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        self.audio  = AudioFeatureExtractor(config)
        
    def encode(self, 
               video_inputs, video_input_lengths,
               audio_inputs, audio_input_lengths,
               targets=None, target_lengths=None, 
               *args, **kwargs):
        features  = self.audio(audio_inputs)
        return features


class VideoConformer(BaseConformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)
        self.video = VisualFeatureExtractor(config)
        
    def encode(self, 
               video_inputs, video_input_lengths,
               audio_inputs, audio_input_lengths,
               targets=None, target_lengths=None, 
               *args, **kwargs):
        features  = self.video(video_inputs, video_input_lengths)
        return features


class TransformerDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(self, config):
        super().__init__()
        decoder = nn.TransformerDecoderLayer(config.decoder.d_model, config.decoder.n_head, 
                                             config.decoder.ff_dim, config.decoder.dropout_p)
        self.decoder = nn.TransformerDecoder(decoder, config.decoder.n_layers)
        
    def forward(self, labels, inputs, train=True, pad_id=None, **kwargs):
        label_mask = self.generate_square_subsequent_mask(labels.shape[1]).to(inputs.device)
        label_pad_mask = self.get_attn_pad_mask(torch.argmax(labels, dim=-1), pad_id) if pad_id else None
        
        labels = labels.permute(1,0,2)
        inputs = inputs.permute(1,0,2)
        
        outputs = self.decoder(labels, inputs, 
                               tgt_mask=label_mask,
                               tgt_key_padding_mask=label_pad_mask)
        
        outputs=outputs.permute(1,0,2)
        return outputs
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.full((sz, sz),-float('inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask
        
    def get_attn_pad_mask(self, seq, pad):
        batch_size, len_seq = seq.size()
        pad_attn_mask = seq.eq(pad)
        return pad_attn_mask
    
    
class FusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(config.encoder.d_model*2, config.encoder.d_model*4),
            nn.BatchNorm1d(config.encoder.d_model*4),
            nn.ReLU(),
            nn.Linear(config.encoder.d_model*4, config.encoder.d_model)
        )
        
    def forward(self, visualFeatures, audioFeatures):
        features = torch.cat([visualFeatures, audioFeatures], dim=-1)
        batch_seq_size = features.shape[:2]
        features = torch.flatten(features, end_dim=1)
        outputs = self.MLP(features)
        outputs = outputs.view(*batch_seq_size, -1)
        return outputs
    
    
class VisualFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.front = VisualFrontEnd(config)
        self.back  = VisualBackEnd(config)
        self.linear = nn.Linear(512, 256)
        
    def forward(self, inputs, input_lengths):
        if self.config.video.use_raw_vid=='on':
            outputs = self.front(inputs)
        else : 
            outputs = inputs
        if 'lip' in self.config.train.transcripts_path_train:
            outputs = self.linear(outputs)
        else:
            pass
        outputs = self.back(outputs, input_lengths)
        return outputs
        
class VisualFrontEnd(nn.Module):
    '''
    input :  (BxCxLxHxW)
    output : (BxLxD)
    '''
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(config.video.n_channels, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3),(1,2,2))
        )
        self.conv2 = get_residual_layer(2, 64, 64, 3, dim=2, identity=True)
        self.conv3 = get_residual_layer(2, 64, 128, 3, dim=2)
        self.conv4 = get_residual_layer(2, 128, 256, 3, dim=2)
        self.conv5 = get_residual_layer(2, 256, config.encoder.d_model, 3, dim=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, inputs):
        outputs = self.conv1(inputs).permute(0,2,1,3,4)
        outputs = outputs.flatten(end_dim=1)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.avg_pool(outputs)
        outputs = outputs.view(inputs.shape[0], inputs.shape[2], -1)
        return outputs
        
class VisualBackEnd(nn.Module):
    '''
    input :  (BxLxD)
    output : (BxLxD)
    '''
    def __init__(self, config):
        super().__init__()
        self.is_transformer = config.encoder.method == 'transformer'
        if config.encoder.method == 'transformer':
            encoder = nn.TransformerEncoderLayer(config.encoder.d_model, config.encoder.n_head, 
                                             config.decoder.ff_dim, config.encoder.dropout_p)
            self.layers = nn.TransformerEncoder(encoder, config.encoder.n_layers)
        else:
            self.conformer = ConformerEncoder(config.encoder.d_model, config.encoder.d_model, 
                                              config.encoder.n_layers, config.encoder.n_head, input_dropout_p=config.encoder.dropout_p)
            self.layers = nn.Sequential(*self.conformer.layers)
        
    def forward(self, inputs, input_lengths):
        self.is_transformer: inputs = inputs.permute(1,0,2)
        outputs = self.layers(inputs)
        self.is_transformer: outputs = outputs.permute(1,0,2)
        return outputs
        
class AudioFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.front = AudioFrontEnd(config)
        self.back  = AudioBackEnd(config)
        
    def forward(self, inputs):
        outputs = self.front(inputs)
        outputs = outputs.permute(0,2,1)
        outputs = self.back(outputs, inputs.size(0))
        return outputs
    
class AudioFrontEnd(nn.Module):
    '''
    input :  (BxCxL)
    output : (BxDxL`) L`:= length of audio sequence with 30 Hz
    '''
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.audio.n_channels, config.encoder.audio.channel_0, 
                      kernel_size=config.encoder.audio.kernel_size, stride=config.encoder.audio.stride, 
                      padding=config.encoder.audio.kernel_size//2), # 80 : 5ms
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = get_residual_layer(2, config.encoder.audio.channel_0, config.encoder.audio.channel_0, 3, identity=True)
        self.conv3 = get_residual_layer(2, config.encoder.audio.channel_0, config.encoder.audio.channel_1, 3)
        self.conv4 = get_residual_layer(2, config.encoder.audio.channel_1, config.encoder.audio.channel_2, 3)
        self.conv5 = get_residual_layer(2, config.encoder.audio.channel_2, config.encoder.d_model, 3)
        #self.avg_pool = nn.AvgPool1d(21, 20, padding=10) # -> 30fps
        self.avg_pool = nn.AvgPool1d(21, 600//config.encoder.fps, padding=10) # -> 60fps
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.avg_pool(outputs)
        return outputs

class AudioBackEnd(nn.Module):
    '''
    input :  (BxLxD)
    output : (BxLxD)
    '''
    def __init__(self, config):
        super().__init__()
        self.is_transformer = config.encoder.method == 'transformer'
        if config.encoder.method == 'transformer':
            encoder = nn.TransformerEncoderLayer(config.encoder.d_model, config.encoder.n_head, 
                                             config.decoder.ff_dim, config.encoder.dropout_p)
            self.layers = nn.TransformerEncoder(encoder, config.encoder.n_layers)
        else:
            self.conformer = ConformerEncoder(config.encoder.d_model, config.encoder.d_model, 
                                              config.encoder.n_layers, config.encoder.n_head, input_dropout_p=config.encoder.dropout_p)
            self.layers = nn.Sequential(*self.conformer.layers)
        
    def forward(self, inputs, input_lengths):
        self.is_transformer: inputs = inputs.permute(1,0,2)
        outputs = self.layers(inputs)
        self.is_transformer: outputs = outputs.permute(1,0,2)
        return outputs
   
   
def get_residual_layer(num_layer, 
                       in_channels, out_channels, 
                       kernel_size, dim=1, identity=False):
    layers = nn.Sequential()
    for i in range(num_layer):
        parameters = [out_channels, out_channels, kernel_size] if i else [in_channels, out_channels, kernel_size, identity]
        layer = ResidualCell(*parameters) if dim==1 else ResidualCell2d(*parameters)
        layers.add_module(f'{i}', layer)
    return layers
    
    
    
class ResidualCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, identity=True):
        super().__init__()
        if identity:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x
    
    
    
class ResidualCell2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, identity=True):
        super().__init__()
        if identity:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            stride = 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x
        
        

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)



class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.l_norm = nn.LayerNorm(dim)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        
        _x = x.clone()
        x = self.l_norm(x)
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h + _x



class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))
    
    
    
class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, train=True):
        super().__init__()
        self.dim = dim
        if train:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        else:
            self.pos_embedding = torch.stack([self.get_angle(pos, torch.arange(dim)) for pos in range(seq_len)])
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding.to(x.device)
            
    def get_angle(self, position, i):
        angles = 1 / torch.float_power(10000, (2 * (i // 2)) / self.dim)
        return position * angles