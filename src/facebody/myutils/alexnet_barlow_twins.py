import torch.nn as nn

# ---------------------------- AlexNetGN Backbone ---------------------------- #
class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)

class AlexNetGN(nn.Module):
    def __init__(self, in_channel=3, out_dim=128, l2norm=True):
        super().__init__()
        self._l2norm = l2norm

        # Conv layers
        conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        ave_pool = nn.AdaptiveAvgPool2d((6, 6))

        # Head layers
        fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc8 = nn.Sequential(
            nn.Linear(4096, out_dim)
        )
        head = [fc6, fc7, fc8]
        if self._l2norm: 
            head.append(Normalize(2))

        self.backbone = nn.Sequential(
            conv_block_1,
            conv_block_2,
            conv_block_3,
            conv_block_4,
            conv_block_5,
            ave_pool,
        )
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return self.head(x)


# --------------------------- AlexNetGN Supervised --------------------------- #å
class AlexNetGNSupervised(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = AlexNetGN(in_channel=3, out_dim=n_classes, l2norm=False)

    def forward(self, x):
        return self.model(x)


# ------------------- AlexNet Barlow Twins Self-Supervised ------------------- #å
class BarlowTwins(nn.Module):
    def __init__(self, backbone, lambd=0.0051, batch_size=2048, projector_sizes=[4096, 4096, 4096]):
        super().__init__()
        self.lambd = lambd
        self.batch_size = batch_size
        self.backbone = backbone
        self.flatten = nn.Flatten()

        # Projector
        sizes = [256*6*6] + projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Normalization layer for representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.flatten(self.backbone(x)))
        return self.bn(z)

class AlexNetBTReadout(nn.Module):
    """Backbone (+ optional projector slice) + linear readout."""
    def __init__(self, backbone: nn.Module, in_features: int, n_classes: int,
                 projector: nn.Sequential | None = None, bias: bool = False):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.projector = projector
        self.readout = nn.Linear(in_features, n_classes, bias=bias)

    def forward(self, x):
        h = self.flatten(self.backbone(x))
        if self.projector is not None:
            h = self.projector(h)
        return self.readout(h)
