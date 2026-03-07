import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from utils.noise_schedules import cosine_noise_schedule

class DDIM(nn.Module):
    def __init__(self,
                 in_channels=3,
                 default_imsize=32,
                 backbone=None,
                 pretrained_backbone=None,
                 noise_schedule=cosine_noise_schedule):
        super().__init__()

        self.in_channels = in_channels
        self.default_imsize = default_imsize
        self.noise_schedule = noise_schedule
        if pretrained_backbone is not None:
            self.backbone = self.pretrained_backbone
        else:
            self.backbone = backbone

    def forward(self, t, x, label=None):
        self.backbone(t, x, label)

    def sample(self, batch_size=1, x=None, label=None, nsteps=20, device=None, breakstep=-1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        self.backbone.eval()
        if x is None:
            x = torch.normal(0, 1, (batch_size, self.in_channels, self.default_imsize, self.default_imsize))

        for i in range(nsteps, 0, -1):
            if i == breakstep:
                return x
            
            t = i*torch.ones(batch_size, device=device)/nsteps
            beta_t = self.noise_schedule(t)
            score = self(t, x, label)

            alpha_t = 1 - beta_t
            beta_t_prev = self.noise_schedule(1 - 1/nsteps)
            alpha_t_prev = 1 - beta_t_prev

            x *= torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None]
            score_correction = (torch.sqrt(beta_t_prev[:, None, None, None]) - torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None] * torch.sqrt(beta_t[:, None, None, None])) * score
            x += score_correction

        return x
    
class EmbeddingModule(nn.Module):
    def __init__(self, emb_dim, conditional=False, num_classes=None):
        super().__init__()

        self.emb_dim = emb_dim
        if conditional:
            self.class_embeddings = nn.Embedding(num_classes, emb_dim)
        
    def forward(self, t, label=None):
        d = self.emb_dim // 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targ = t[:, None] / (10000 ** (torch.arange(d, device=device)) / (d - 1))[None, :]
        emb = torch.cat((torch.sin(targ), torch.cos(targ)), dim=1)

        if self.conditional:
            emb += self.class_embeddings(label)

        return emb
    
class MinimalResnet:
    pass

class MinimalUNet(nn.Module):
    def __init__(self,
                 k=3,
                 channels=3,
                 fchannels=[64,128,256],
                 ublock_depth=2,
                 emb_dim=256,
                 padding_mode='zeros',
                 conditional=False,
                 num_classes=None):
        super().__init__()

        self.fchannels = fchannels
        self.channels = channels
        self.conditional = conditional
        self.emb_dim = emb_dim
        self.k = k

        self.embedding = EmbeddingModule(emb_dim=emb_dim, conditional=conditional, num_classes=num_classes)

        in_channels = channels
        self.encoder_blocks = nn.ModuleList()
        for f in fchannels[:-1]:
            self.encoder_blocks.append(UBlock(in_features=in_channels, out_features=f, k=k, padding_mode=padding_mode, depth=ublock_depth, emb_dim=emb_dim))
            in_channels = f
        
        self.bottleneck = UBlock(in_features=fchannels[-2], out_features=fchannels[-1], k=k, padding_mode=padding_mode, depth=ublock_depth, emb_dim=emb_dim)

        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(fchannels)-1, 0, -1):
            in_channels = fchannels[i]
            out_channels = fchannels[i-1]
            self.upsample_blocks.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2))
            # 2x in_channels due to concatenation of the skip connection from the corresponding encoder block along the channel axis.
            self.decoder_blocks.append(UBlock(in_features=2 * in_channels, out_features=out_channels, k=k, padding_mode=padding_mode, depth=ublock_depth, emb_dim=emb_dim))

        self.last_emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, fchannels[0]))
        self.output_conv = nn.Conv2d(in_channels=fchannels[0], out_channels=channels, kernel_size=1, padding='same', padding_mode=padding_mode)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, t, x, label=None):
        embedding_vec = self.embedding(t, label=label)

        skip_connections = []

        for encoder in self.encoder_blocks:
            x = encoder(x, embedding_vec)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, embedding_vec)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoder_blocks)):
            upconv = self.upsample_blocks[i](x)
            skip = skip_connections[i]
            x = torch.cat((skip,upconv), dim=1)
            x = self.decoder_blocks[i](x, embedding_vec)

        return self.output_conv(x + self.last_emb(embedding_vec)[:, :, None, None])

class UBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 k=3,
                 padding_mode='zeros',
                 depth=2,
                 emb_dim=32):
        super().__init__()

        self.emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, in_features))

        module_list = []
        for i in range(depth):
            if i == 0:
                x = in_features
            else:
                x = out_features

            module_list.append(nn.Conv2d(x, out_features, kernel_size=k, padding='same', padding_mode=padding_mode))
            module_list.append(nn.ReLU())

        self.model = nn.Sequential(*module_list)

    def forward(self, x, embedding):
        return self.model(x + self.emb(embedding)[:, :, None, None])