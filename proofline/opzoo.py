"""A zoo of real operators, each a plain nn.Module (B,C,H,W) -> (B,C,H,W)."""
import torch, torch.nn as nn, torch.nn.functional as F

class DenseConv(nn.Module):
    def __init__(s, c, k=3, depth=4):
        super().__init__()
        s.net = nn.Sequential(*[nn.Sequential(nn.Conv2d(c,c,k,padding=k//2), nn.GELU()) for _ in range(depth)])
    def forward(s,x): return s.net(x)

class DWConv(nn.Module):
    def __init__(s, c, k=3, depth=4):
        super().__init__()
        s.net = nn.Sequential(*[nn.Sequential(nn.Conv2d(c,c,k,padding=k//2,groups=c), nn.GELU()) for _ in range(depth)])
    def forward(s,x): return s.net(x)

class MHABlock(nn.Module):
    def __init__(s, c, heads=8):
        super().__init__()
        s.h=heads; s.n=nn.LayerNorm(c); s.qkv=nn.Linear(c,3*c); s.o=nn.Linear(c,c)
    def forward(s,x):
        B,C,H,W=x.shape
        t=s.n(x.flatten(2).transpose(1,2))
        q,k,v=s.qkv(t).chunk(3,-1)
        q=q.reshape(B,-1,s.h,C//s.h).transpose(1,2)
        k=k.reshape(B,-1,s.h,C//s.h).transpose(1,2)
        v=v.reshape(B,-1,s.h,C//s.h).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v)
        a=a.transpose(1,2).reshape(B,H*W,C)
        return x + s.o(a).transpose(1,2).reshape(B,C,H,W)

class MLPMixer(nn.Module):
    def __init__(s, c, depth=4):
        super().__init__()
        s.net = nn.Sequential(*[nn.Sequential(nn.LayerNorm(c), nn.Linear(c,4*c), nn.GELU(), nn.Linear(4*c,c)) for _ in range(depth)])
    def forward(s,x):
        B,C,H,W=x.shape
        t=x.flatten(2).transpose(1,2)
        t=s.net(t)
        return t.transpose(1,2).reshape(B,C,H,W)

class WindowAttn(nn.Module):
    def __init__(s, c, win=7, heads=8, depth=2):
        super().__init__()
        s.win=win; s.h=heads
        s.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(c), nn.Linear(c,3*c), nn.Linear(c,c)]) for _ in range(depth)])
    def forward(s,x):
        B,C,H,W=x.shape; w=s.win
        pad=(w-H%w)%w, (w-W%w)%w
        xp=F.pad(x,(0,pad[1],0,pad[0]))
        Hp,Wp=xp.shape[-2:]
        for norm,qkv,proj in s.blocks:
            t=norm(xp.permute(0,2,3,1))
            q,k,v=qkv(t).chunk(3,-1)
            q=q.reshape(B,Hp//w,w,Wp//w,w,-1).permute(0,1,3,2,4,5)
            k=k.reshape(B,Hp//w,w,Wp//w,w,-1).permute(0,1,3,2,4,5)
            v=v.reshape(B,Hp//w,w,Wp//w,w,-1).permute(0,1,3,2,4,5)
            a=F.scaled_dot_product_attention(q,k,v)
            a=a.permute(0,1,3,2,4,5).reshape(B,Hp,Wp,C)
            xp=xp+proj(a).permute(0,3,1,2)
        return xp[:,:,:H,:W]

class MambaSSM(nn.Module):
    """diagonal selective SSM, chunked scan (parallel within chunk, recurrent across).
    A realistic proxy for a fused Mamba mixer, no custom CUDA needed."""
    def __init__(s, c, d_state=16, expand=2, chunk=32, depth=2):
        super().__init__()
        s.d=c; s.n=d_state; s.e=expand; s.ch=chunk; s.depth=depth
        di=s.d*s.e
        s.in_proj=nn.Linear(c, 2*di)
        s.conv=nn.Conv1d(di,di,4,padding=3,groups=di)
        s.x_proj=nn.Linear(di, d_state*2+1)
        s.dt_proj=nn.Linear(1, di)
        s.A=nn.Parameter(torch.randn(di,d_state)*0.01 - 1.0)
        s.out_proj=nn.Linear(di,c)
    def forward(s,x):
        B,C,H,W=x.shape
        u=x.flatten(2).transpose(1,2)
        for _ in range(s.depth):
            z,res=s.in_proj(u).chunk(2,-1)
            z=s.conv(z.transpose(1,2))[:,:,:u.shape[1]].transpose(1,2)
            z=F.silu(z)
            BC=s.x_proj(z); delta=BC[...,0:1]; Bm=BC[...,1:1+s.n]; Cm=BC[...,1+s.n:]
            dt=F.softplus(s.dt_proj(delta))
            A=torch.exp(dt.unsqueeze(-1)*s.A)
            # chunked scan
            L=u.shape[1]; outs=[]
            h=torch.zeros(B, z.shape[-1], s.n, device=z.device)
            for st in range(0,L,s.ch):
                en=min(st+s.ch,L); Lc=en-st
                Ac=A[:,st:en]; Bc=Bm[:,st:en]; Cc=Cm[:,st:en]; zc=z[:,st:en]
                acc=torch.zeros(B,Lc,z.shape[-1],s.n,device=z.device)
                cur=torch.zeros(B,z.shape[-1],s.n,device=z.device)
                for t in range(Lc):
                    cur=Ac[:,t]*cur + (zc[:,t].unsqueeze(-1)*Bc[:,t].unsqueeze(-2))
                    acc[:,t]=cur
                h=h+acc[:,-1]
                outs.append((acc*Cc.unsqueeze(-2)).sum(-1))
            y=torch.cat(outs,1)*F.silu(res)
            u=u+s.out_proj(y)
        return u.transpose(1,2).reshape(B,C,H,W)

ZOO = {
 "dense_conv x4":  lambda c: DenseConv(c,3,4),
 "dwconv x4":      lambda c: DWConv(c,3,4),
 "mha block x2":   lambda c: nn.Sequential(MHABlock(c), MHABlock(c)),
 "sdpa attn x2":   lambda c: nn.Sequential(MHABlock(c), MHABlock(c)),
 "mlp_mixer x4":   lambda c: MLPMixer(c,4),
 "window_attn x2": lambda c: WindowAttn(c,7,8,2),
 "mamba_ssm x2":   lambda c: MambaSSM(c,16,2,32,2),
}
