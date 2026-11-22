"""
Back-calibration: turn the floor into a latency budget.

Two models, both fitted from (traced analysis, measured latency) pairs.

  simple      T = a0 + tau * n_ops             2 parameters  <- default
  category    T = a0 + sum_k c_k * n_k         9 parameters  <- needs more data

Why not always the category model: fitting 9 costs from 4-5 architectures
overfits.  Measured leave-one-architecture-out error (train on 4, predict the
unseen 5th, same backbone):

    simple    median 1.16x   mean 1.18x   max 1.32x
    category  median 1.12x   mean 1.21x   max 1.72x

The category model has a better median but worse tail; with 4 training points
its extra parameters buy variance, not accuracy.  Use it when you can measure
>=10 architectures.
"""
import json
from dataclasses import replace
from .devices import GPU

CATEGORIES = ("matmul","conv","norm","reduce","layout","pool","act","elem")
_RULES = {
 "matmul": ("mm.","bmm.","matmul","addmm","baddbmm","sdpa","scaled_dot"),
 "conv":   ("convolution","conv2d","_convolution"),
 "norm":   ("layer_norm","group_norm","batch_norm","rms_norm"),
 "reduce": ("sum.","mean.","amax","argmax","cumsum"),
 "layout": ("clone","permute","view","reshape","contiguous","expand","unfold",
            "copy","_to_copy","transpose","narrow","slice","as_strided"),
 "pool":   ("avg_pool","max_pool","pool2d","adaptive"),
 "act":    ("relu","gelu","silu","sigmoid","tanh","exp.","softmax","clamp","where","neg"),
}

def categorize(rec):
    """rec: analysis["rec"]  ->  counts per category"""
    out = {c: 0 for c in CATEGORIES}
    for name, v in rec.items():
        short = name.split(".")[1] if "." in name else name
        hit = "elem"
        for c, keys in _RULES.items():
            if any(k in short for k in keys):
                hit = c; break
        out[hit] += v["count"]
    return out

def _base(a, gpu):
    return max(a["W_GFLOP"]*1e9/gpu.F, a["Q_MB"]*1e6/gpu.BW)

def fit_simple(pairs, gpu):
    """pairs: [(analysis, T_measured_s)] -> (gpu with t, a0)"""
    xs = [float(a["n_ops"]) for a,_ in pairs]
    ys = [float(tm - _base(a,gpu)) for a,tm in pairs]
    n = len(xs)
    if n == 0: return gpu, 0.0
    if n == 1:
        tau = max(gpu.t, ys[0]/xs[0]) if xs[0] else gpu.t
        return replace(gpu, t=tau), 0.0
    sx,sy = sum(xs),sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
    den = n*sxx - sx*sx
    tau = (n*sxy - sx*sy)/den if den else max(gpu.t, sy/max(sx,1))
    a0  = (sy - tau*sx)/n
    if tau <= 0: tau, a0 = max(gpu.t, sy/max(sx,1)), 0.0
    return replace(gpu, t=tau), a0

def fit_category(pairs, gpu):
    """Non-negative least squares on per-category op counts.
    Returns (costs dict, a0).  Requires >= 6 measured models to be meaningful."""
    import numpy as np
    A=[]; b=[]
    for a, tm in pairs:
        cnt = categorize(a["rec"])
        A.append([1.0]+[cnt[c] for c in CATEGORIES])
        b.append(float(tm - _base(a,gpu)))
    A=np.array(A,float); b=np.array(b,float)
    try:
        from scipy.optimize import nnls
        w,_ = nnls(A,b)
    except Exception:
        w = np.maximum(np.linalg.lstsq(A,b,rcond=None)[0], 0.0)
    w = np.maximum(w, 0.0)
    return dict(zip(CATEGORIES, w[1:])), float(w[0])

def predict_s(analysis, gpu, a0=0.0):
    return _base(analysis, gpu) + a0 + analysis["n_ops"]*gpu.t

def predict_us(analysis, gpu, a0=0.0):
    return predict_s(analysis, gpu, a0)*1e6

def predict_category(analysis, costs, a0, gpu):
    cnt = categorize(analysis["rec"])
    return _base(analysis,gpu) + a0 + sum(costs[c]*cnt[c] for c in CATEGORIES)

def save_profile(gpu, a0, path, costs=None):
    d={"name":gpu.name,"F":gpu.F,"BW":gpu.BW,"t":gpu.t,"a0":a0}
    if costs: d["costs"]=costs
    with open(path,"w") as f: json.dump(d,f,indent=2)
    return path

def load_profile(path):
    with open(path) as f: d=json.load(f)
    return GPU(d["name"],d["F"],d["BW"],d["t"]), float(d.get("a0",0.0)), d.get("costs")
