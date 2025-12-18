import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F


class MultiZAE(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(MultiZAE, self).__init__()
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
        self.lambda_latent = 0.1

        # server gửi xuống prototype các nhãn != 0
        self.attack_prototypes = []  # list[tensor(D)]

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return z, x_hat

    def load_server_prototypes(self, proto_dict):#lấy hết các prototype từ server !=0: attack_prototypes
        """
        proto_dict: {label: tensor(latent_dim)} với label != 0
        """
        self.attack_prototypes = [
            v.detach().clone() for k, v in proto_dict.items() if k != 0
        ]

    # ================================
    # ✅ Softmax-weighted target Z=1
    # ================================
    def weighted_attack_target(self, device):
        """
        Tính Z_target = sum_k softmax(-||Zk-1||^2) * Zk
        """
        if len(self.attack_prototypes) == 0:
            return torch.ones(1, device=device)

        P = torch.stack(self.attack_prototypes).to(device)  
        one = torch.ones_like(P)

        dist = ((P - one) ** 2).sum(dim=1)  # (K,)
        weights = torch.softmax(-dist, dim=0)  # (K,)

        Z_target = (weights.unsqueeze(1) * P).sum(dim=0)  # (D,)
        return Z_target 

    # ================================
    # ✅ Binary Latent Alignment Loss
    # ================================
    def binary_latent_loss(self, z, y, loss_type="mse"):
        """
        loss_type:
            - "mse"     : ép độ lớn + hướng (Euclidean)
            - "cosine"  : ép HƯỚNG (góc)
        
        y=0  -> ép z → 0
        y!=0 -> ép z → Z_target (softmax weighted)
        """
        device = z.device
        Z_target = self.weighted_attack_target(device)  

        y_bin = (y != 0).float().unsqueeze(1)  # (B,1)

        # Target latent
        target = y_bin * Z_target + (1 - y_bin) * 0.0  # (B,D)

        # =========================
        # ✅ CHỌN KIỂU LOSS
        # =========================
        if loss_type == "mse":
            # MSE Loss (ép cả độ lớn + hướng)
            latent_loss = ((z - target) ** 2).mean()

        elif loss_type == "cosine":
            # Cosine Loss (chỉ ép hướng)
            cos_sim = F.cosine_similarity(z, target, dim=1)
            latent_loss = (1 - cos_sim).mean()

        else:
            raise ValueError("loss_type phải là 'mse' hoặc 'cosine'")

        return latent_loss

    # ================================
    # ✅ Tổng loss cuối cùng
    # ================================
    def calculate_loss(self, x, y, latent_loss_type="mse"):
        z, x_hat = self.forward(x)

        re_loss = self.loss_function(x_hat, x)

        latent_loss = self.binary_latent_loss(
            z, y, loss_type=latent_loss_type
        )

        total_loss = re_loss + self.lambda_latent * latent_loss
        return total_loss, re_loss, latent_loss
