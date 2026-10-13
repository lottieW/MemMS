import torch
from torch import nn
from .memory import NormalMemory

class TemporalBlock(nn.Module):
    def __init__(self, h, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(h,h,3,padding=dilation,dilation=dilation),
                                 nn.GELU(),nn.Dropout(dropout),nn.Conv1d(h,h,1))
    def forward(self,x):
        return x+self.net(x)

class base_Model(nn.Module):
    """Input and reconstruction: [batch, time, variables], matching DuSK's model boundary."""
    def __init__(self, configs, device=None):
        super().__init__()
        c = self.c = configs
        h = c.d_model
        self.variable_embedding = nn.Embedding(c.input_channels,h)
        self.encoder = nn.Sequential(nn.Conv1d(1,h,3,padding=1),nn.GELU(),
            *[TemporalBlock(h,2**i,c.dropout) for i in range(c.tcn_layers)])
        self.experts = nn.ModuleList([TemporalBlock(h,2**i,c.dropout) for i in range(c.num_experts)])
        self.variable_attention = nn.MultiheadAttention(h, 1, dropout=c.dropout, batch_first=True)
        self.variable_norm = nn.LayerNorm(h)
        self.temporal_pool = nn.Linear(c.window_size, 1)
        self.key_projection = nn.Sequential(nn.Linear(2*h,h), nn.GELU(), nn.Linear(h,h))
        self.router = nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.Linear(h,c.num_experts))
        self.decoder = nn.Sequential(nn.Linear(2*h,h),nn.GELU(),nn.Linear(h,c.window_size))
        self.memory = NormalMemory(c)
        self.register_buffer('score_mean',torch.zeros(2))
        self.register_buffer('score_std',torch.ones(2))

    def forward(self,x):
        b,l,d = x.shape
        if l != self.c.window_size or d != self.c.input_channels:
            raise ValueError('Input must be [B, window_size, input_channels]')
        temporal = self.encoder(x.permute(0,2,1).reshape(b*d,1,l))
        emb = self.variable_embedding.weight.unsqueeze(0).expand(b,-1,-1)
        temporal = temporal + emb.reshape(b*d,-1,1)
        z = self.temporal_pool(temporal).squeeze(-1).reshape(b,d,-1)
        context, _ = self.variable_attention(z, z, z, need_weights=False)
        z = self.variable_norm(z + context)
        global_z = self.key_projection(torch.cat([z.mean(1), z.amax(1)], -1))
        current = self.router(torch.cat([z,global_z[:,None].expand_as(z)],-1)).softmax(-1)
        prior,confidence,novelty = self.memory.query(global_z)
        strength = self.c.memory_strength
        routes = (current.clamp_min(1e-8).log()+strength*confidence[:,None,None]*prior.clamp_min(1e-8).log()).softmax(-1)
        experts = torch.stack([self.temporal_pool(e(temporal)).squeeze(-1).reshape(b,d,-1) + z for e in self.experts],2)
        fused = (routes.unsqueeze(-1)*experts).sum(2)
        recon = self.decoder(torch.cat([fused,emb],-1)).permute(0,2,1)
        return dict(x_recon=recon, z_global=global_z, current=current, routes=routes,
                    prior=prior, confidence=confidence, novelty=novelty, experts=experts)
