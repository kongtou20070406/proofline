"""Measure the TRUE critical path (longest dependency chain) via torch.fx."""
import torch, torch.fx as fx

def critical_path(model, example):
    """Return (n_nodes, longest_chain_len) for the traced graph."""
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        return None, None, repr(e)[:60]
    # topological order is gm.graph.nodes order (fx emits in exec order)
    depth = {}
    for n in gm.graph.nodes:
        d = 1
        for inp in n.all_input_nodes:
            d = max(d, depth.get(inp, 0) + 1)
        depth[n] = d
    nodes = [n for n in gm.graph.nodes if n.op not in ("placeholder","output")]
    return len(nodes), (max(depth.values()) if depth else 0), None
