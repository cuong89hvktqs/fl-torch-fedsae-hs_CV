import torch
import torch.nn as nn
from math import sqrt



class MultiLossAE(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(MultiLossAE, self).__init__()
        self.epsilon = epsilon
        
        self.enc = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),
        )
        self.dec = nn.Sequential(
            nn.Linear(round(sqrt(n_features)) + 1, round(n_features * 0.5)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),
            nn.Tanh(),
            nn.Linear(round(n_features * 0.75), n_features),
        )
        self.loss_function = nn.MSELoss()
        self.lambda_latent =0.1 #lambda_latent
        self.alpha_proto = 0.1#alpha_proto
        self.prototypes = {}  # dict[label] = tensor(latent_dim)
    
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return encode, decode
    
    def load_server_prototypes(self, proto_dict):
        """
        Nhận dictionary prototype từ server.
        proto_dict: {label: tensor(latent_dim)}
        """
        self.prototypes = {lab: vec.detach().clone() for lab, vec in proto_dict.items()}
    
    def prototype_contrastive_loss(self, z, y):
        if len(self.prototypes) == 0:
            return torch.tensor(0.0, device=z.device)

        proto_list, id_list = [], []
        for lab, p in self.prototypes.items():
            proto_list.append(p.to(z.device))
            id_list.append(lab)

        proto_mat = torch.stack(proto_list)          # (C, D)
        labels = torch.tensor(id_list).to(z.device)  # (C,)

        B, D = z.shape
        C = proto_mat.shape[0]

        z_expand = z.unsqueeze(1)         # (B,1,D)
        p_expand = proto_mat.unsqueeze(0) # (1,C,D)
        dist = ((z_expand - p_expand) ** 2).sum(dim=2)  # (B,C)

        pos_mask = (y.unsqueeze(1) == labels.unsqueeze(0))  # (B,C)
        if pos_mask.sum() > 0:
            pos_dist = dist[pos_mask]
            L_pos = pos_dist.mean()
        else:
            L_pos = torch.tensor(0.0, device=z.device)

        neg_mask = ~pos_mask
        neg_dist = dist[neg_mask]
        if neg_dist.numel() > 0:
            L_neg = (1.0 / (neg_dist + self.epsilon)).mean()
        else:
            L_neg = torch.tensor(0.0, device=z.device)

        return L_pos + self.alpha_proto * L_neg
    
    def calculate_loss(self, x, y):
        encode, decode = self.forward(x)
        re_loss = self.loss_function(decode, x)
        L_proto = self.prototype_contrastive_loss(encode, y)
        return re_loss + self.lambda_latent  * L_proto
