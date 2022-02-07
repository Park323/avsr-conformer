import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.conformer.encoder import ConformerEncoder

'''
Video : 30 fps
Audio : 16000 Hz
'''

class AudioVisualConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.decoder.vocab_size
        self.fusion = FusionModule(config)
        self.visual = VisualFeatureExtractor(config)
        self.audio  = AudioFeatureExtractor(config)
        self.target_embedding = nn.Linear(self.vocab_size, config.decoder.d_model)
        self.decoder= TransformerDecoder(config)
        self.ceLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        self.ctcLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                targets, target_lengths, 
                *args, **kwargs):
        audio_inputs = audio_inputs[:, :, :video_inputs.size(2)*480]
        visualFeatures = self.visual(video_inputs)
        audioFeatures  = self.audio(audio_inputs)
        features = torch.cat([visualFeatures, audioFeatures], dim=-1)
        outputs = self.fusion(features)
        targets = F.one_hot(targets, num_classes = self.vocab_size)
        targets = self.target_embedding(targets.to(torch.float32))
        att_out = F.log_softmax(self.ceLinear(
            self.decoder(targets, outputs)
            ), dim=-1)
        ctc_out = F.log_softmax(self.ctcLinear(outputs), dim=-1)
        return (att_out, ctc_out)

class AudioConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.decoder.vocab_size
        self.audio  = AudioFeatureExtractor(config)
        self.target_embedding = nn.Linear(self.vocab_size, config.decoder.d_model)
        self.decoder= TransformerDecoder(config)
        self.ceLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        self.ctcLinear = nn.Linear(config.decoder.d_model, self.vocab_size)
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                targets, target_lengths, 
                *args, **kwargs):
        audio_inputs = audio_inputs
        audioFeatures  = self.audio(audio_inputs)
        targets = F.one_hot(targets, num_classes = self.vocab_size)
        targets = self.target_embedding(targets.to(torch.float32))
        att_out = F.log_softmax(self.ceLinear(
            self.decoder(targets, audioFeatures)
            ), dim=-1)
        ctc_out = F.log_softmax(self.ctcLinear(audioFeatures), dim=-1)
        return (att_out, ctc_out)
    
class TransformerDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(self, config):
        super().__init__()
        decoder = nn.TransformerDecoderLayer(config.decoder.d_model, config.decoder.n_head, 
                                             config.decoder.ff_dim, config.decoder.dropout_p, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder, config.decoder.n_layers)
    def forward(self, labels, inputs):
        label_mask = torch.zeros((labels.shape[1], labels.shape[1])).to(inputs.device)
        for i in range(labels.shape[1]):
            label_mask[i, i+1:]=1.
        outputs = self.decoder(labels, inputs, label_mask)
        return outputs
    
class FusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(config.encoder.d_model*2, config.encoder.d_model*4),
            nn.BatchNorm1d(config.encoder.d_model*4),
            nn.ReLU(),
            nn.Linear(config.encoder.d_model*4, config.encoder.d_model)
        )
        
    def forward(self, features):
        batch_seq_size = features.shape[:2]
        features = torch.flatten(features, end_dim=1)
        outputs = self.MLP(features)
        outputs = outputs.view(*batch_seq_size, -1)
        return outputs
    
class VisualFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.front = VisualFrontEnd(config)
        self.back  = VisualBackEnd(config)
        
    def forward(self, inputs):
        outputs = self.front(inputs)
        outputs = self.back(outputs, inputs.size(0))
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
        self.conformer = ConformerEncoder(config.encoder.d_model, config.encoder.d_model, 
                                          config.encoder.n_layers, config.encoder.n_head, input_dropout_p=config.encoder.dropout_p)
        
    def forward(self, inputs, input_lengths):
        outputs, _ = self.conformer(inputs, input_lengths)
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
            nn.Conv1d(config.audio.n_channels, 64, kernel_size=79, stride=3, padding=39), # 80 : 5ms
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = get_residual_layer(2, 64, 64, 3, identity=True)
        self.conv3 = get_residual_layer(2, 64, 128, 3)
        self.conv4 = get_residual_layer(2, 128, 256, 3)
        self.conv5 = get_residual_layer(2, 256, config.encoder.d_model, 3)
        self.avg_pool = nn.AvgPool1d(21, 20, padding=10) # -> 30fps
        
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
        self.conformer = ConformerEncoder(config.encoder.d_model, config.encoder.d_model, 
                                          config.encoder.n_layers, config.encoder.n_head, input_dropout_p=config.encoder.dropout_p)
        
    def forward(self, inputs, input_lengths):
        outputs, _ = self.conformer(inputs, input_lengths)
        return outputs
        
   
   
def get_residual_layer(num_layer, in_channels, out_channels, kernel_size, dim=1, identity=False):
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