import torch
import torch.nn as nn

def get_groups(ch): return next((g for g in [8,6,4,3,2,1] if ch % g == 0), 1)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        g = get_groups(out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()
        self.skip = nn.Sequential() if stride==1 and in_ch==out_ch else nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False), nn.GroupNorm(g, out_ch))
    def forward(self, x): return self.act(self.gn2(self.conv2(self.act(self.gn1(self.conv1(x))))) + self.skip(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b = nn.Sequential(ResBlock3D(in_ch, out_ch), ResBlock3D(out_ch, out_ch))
        self.d = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
    def forward(self, x): s = self.b(x); return self.d(s), s

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.u = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.b = nn.Sequential(ResBlock3D(in_ch + skip_ch, out_ch), ResBlock3D(out_ch, out_ch))
    def forward(self, x, s): return self.b(torch.cat([self.u(x), s], dim=1))

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, order=6, out_channels=256):
        super().__init__()
        self.order = order
        self.mlp = nn.Sequential(nn.Linear(4*order, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, out_channels), nn.GELU())
    def forward(self, dab, dbc):
        t_ab, t_bc = dab/365.0, dbc/365.0
        feats =[]
        for k in range(self.order):
            f = (2**k)*torch.pi
            feats.extend([torch.sin(f*t_ab), torch.cos(f*t_ab), torch.sin(f*t_bc), torch.cos(f*t_bc)])
        return self.mlp(torch.stack(feats, dim=1))

class ConvLSTMCell3D(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv3d(in_ch + hidden_ch, 4*hidden_ch, 3, padding=1)
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        C = self.hidden_ch
        i, f, o, g = torch.sigmoid(gates[:,0:C]), torch.sigmoid(gates[:,C:2*C]), torch.sigmoid(gates[:,2*C:3*C]), torch.tanh(gates[:,3*C:4*C])
        c_n = f*c + i*g
        return o*torch.tanh(c_n), c_n

class TempVSNet(nn.Module):
    def __init__(self, base_ch=24):
        super().__init__()
        self.enc_a1 = DownBlock(2, base_ch)
        self.enc_a2 = DownBlock(base_ch, base_ch*2)
        self.enc_a3 = DownBlock(base_ch*2, base_ch*4)
        self.enc_b1 = DownBlock(3, base_ch)
        self.enc_b2 = DownBlock(base_ch, base_ch*2)
        self.enc_b3 = DownBlock(base_ch*2, base_ch*4)
        self.btn_a = nn.Sequential(ResBlock3D(base_ch*4, base_ch*4), nn.Dropout3d(0.15))
        self.btn_b = nn.Sequential(ResBlock3D(base_ch*4, base_ch*4), nn.Dropout3d(0.15))
        self.time_enc = SinusoidalTimeEncoding(6, base_ch*4)
        self.time_prj = nn.Linear(base_ch*4, base_ch*4)
        self.lstm = ConvLSTMCell3D(base_ch*4, base_ch*4)
        self.lstm_prj = nn.Sequential(nn.Conv3d(base_ch*4, base_ch*4, 1), nn.GroupNorm(get_groups(base_ch*4), base_ch*4), nn.GELU())
        self.dec3 = UpBlock(base_ch*4, base_ch*4, base_ch*2)
        self.dec2 = UpBlock(base_ch*2, base_ch*2, base_ch)
        self.dec1 = UpBlock(base_ch,   base_ch,   base_ch)
        self.head = nn.Conv3d(base_ch, 1, 1)

    def forward(self, img_a, mask_a, img_b, mask_b, diff_ab, dab, dbc):
        B = img_a.shape[0]
        xa, sa1 = self.enc_a1(torch.cat([img_a, mask_a], dim=1))
        xa, sa2 = self.enc_a2(xa)
        xa, sa3 = self.enc_a3(xa)
        za = self.btn_a(xa)
        xb, sb1 = self.enc_b1(torch.cat([img_b, mask_b, diff_ab], dim=1))
        xb, sb2 = self.enc_b2(xb)
        xb, sb3 = self.enc_b3(xb)
        zb = self.btn_b(xb)
        t_sp = self.time_prj(self.time_enc(dab, dbc)).view(B, -1, 1, 1, 1)
        za, zb = za + t_sp, zb + t_sp
        h = torch.zeros(B, za.size(1), *za.shape[2:]).to(za.device)
        c = torch.zeros_like(h)
        h, c = self.lstm(za, h, c)
        h, c = self.lstm(zb, h, c)
        x = self.dec3(self.lstm_prj(h), sb3)
        x = self.dec2(x, sb2)
        x = self.dec1(x, sb1)
        return self.head(x)