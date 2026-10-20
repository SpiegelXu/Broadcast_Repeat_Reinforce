import math
import torch
import torch.nn.functional as F

def compute_shifted_c_gamma_CA(A: torch.Tensor):
    """ A ≈ 1 c^T + gamma P + E。"""
    if A.ndim < 2:
        raise ValueError("A must have shape [..., M, M].")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError(f"A must be square, but got {tuple(A.shape[-2:])}.")

    A = A.detach().float()
    M, device, dtype = A.shape[-1], A.device, A.dtype
    if M < 2:
        raise ValueError("Sequence length M must be at least 2.")

    # P[i, i-1] = 1
    P = torch.zeros(M, M, device=device, dtype=dtype)
    rows = torch.arange(1, M, device=device)
    P[rows, rows - 1] = 1.0

    # exclude P to calculate c
    denominator = M - P.sum(dim=0)
    c = (A * (1.0 - P)).sum(dim=-2) / denominator

    # residual、gamma and complete E
    stripe_residual = A - c.unsqueeze(-2)
    gamma = (stripe_residual * P).sum(dim=(-2, -1)) / P.square().sum()
    E = A - c.unsqueeze(-2) - gamma[..., None, None] * P

    return {"c": c, "gamma": gamma, "E": E, "P": P}

def detect_anchors(
    A: torch.Tensor,
    window_size: int = None,
    q_threshold: float = 0.30,
    tail_fraction: float = 0.25,
    tail_margin: float = 0.40,
    is_mask_key: torch.Tensor = None,
):
    A = A.float()
    M = A.shape[-1]
    device = A.device

    if window_size is None:
        window_size = max(2, math.ceil(math.sqrt(M)))

    # exclude shifted diagonal
    A_masked = A.clone()
    rows = torch.arange(1, M, device=device)
    A_masked[..., rows, rows - 1] = float("nan")

    # set window
    padded_A = F.pad(
        A_masked,
        (window_size, window_size),
        value=float("nan"),
    )

    windows = padded_A.unfold(
        dimension=-1,
        size=2 * window_size + 1,
        step=1,
    )


    center_mask = torch.zeros(
        2 * window_size + 1,
        dtype=torch.bool,
        device=device,
    )
    center_mask[window_size] = True

    windows = windows.masked_fill(
        center_mask,
        float("nan"),
    )

    # window median
    background = torch.nanmedian(
        windows,
        dim=-1,
    ).values

    '''
    q25 = torch.nanquantile(
        windows,
        q=tail_fraction,
        dim=-1,
    )

    bottom25_mask = windows <= q25.unsqueeze(-1)

    bottom25_values = windows.masked_fill(
        ~bottom25_mask,
        float("nan"),
    )

    bottom25_mean = torch.nanmean(
        bottom25_values,
        dim=-1,
    )
    
    required_attention = (
        background
        + tail_margin * bottom25_mean
    )
    '''
    
    required_attention = (
            background
            + tail_margin * background
        )

    
    D = (
        torch.log(A.clamp_min(1e-8))
        -
        torch.log(required_attention.clamp_min(1e-8))
    )

    # shifted diagonal is exluded from query's Q30
    D[..., rows, rows - 1] = float("nan")

    
    q30 = torch.nanquantile(
        D,
        q=q_threshold,
        dim=-2,
    )

    anchor_mask = q30 > 0

    if is_mask_key is not None:
        
        is_mask_key = is_mask_key.to(
            device=device,
            dtype=torch.bool,
        )

        mask_shape = (
            [1] * (anchor_mask.ndim - 1)
            + [M]
        )

        anchor_mask &= (
            ~is_mask_key
        ).reshape(mask_shape)

    return anchor_mask

def compute_prototypes(
    K: torch.Tensor,
    anchor_mask: torch.Tensor,
    c: torch.Tensor,
    min_anchors: int = 2,
):
    """ΔK、C_A、C_O、K_A、K_O, head """
    K, c = K.float(), c.float()
    w_A = (anchor_mask.float() * c).unsqueeze(-1)
    w_O = ((~anchor_mask).float() * c).unsqueeze(-1)

    C_A = w_A.sum(dim=-2).squeeze(-1)
    C_O = w_O.sum(dim=-2).squeeze(-1)
    K_A = (K * w_A).sum(dim=-2) / C_A.unsqueeze(-1).clamp_min(1e-8)
    K_O = (K * w_O).sum(dim=-2) / C_O.unsqueeze(-1).clamp_min(1e-8)

    valid_heads = (
        (anchor_mask.sum(dim=-1) >= min_anchors)
        & (C_A > 1e-8)
        & (C_O > 1e-8)
    )
    return K_A - K_O, C_A, C_O, K_A, K_O, valid_heads

def analyze_anchor_transitions(
    Q: torch.Tensor,
    K: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    scopes: dict = None,
    mode: str = "dynamic",
    window_size: int = None,
    q_threshold: float = 0.30,
    min_anchors: int = 2,
    tail_fraction: float = 0.25,
    tail_margin: float = 0.10,
) -> dict:
    #  calculate Transition Gain
    Q, K, A, c = Q.float(), K.float(), A.float(), c.float()
    device, M, d_k = Q.device, Q.shape[-2], Q.shape[-1]


    scopes = dict(scopes or {})
    scopes.setdefault("all", torch.ones(M, dtype=torch.bool, device=device))

    anchor_mask = detect_anchors(
        A,
        window_size,
        #q_threshold,
        tail_fraction=tail_fraction,
        tail_margin=tail_margin,
        is_mask_key=scopes.get("is_mask_key"),
    )
    delta_K, C_A, C_O, K_A, K_O, valid_heads = compute_prototypes(
        K, anchor_mask, c, min_anchors
    )

    g = torch.einsum("lhmd,lhd->lhm", Q, delta_K)

    if mode == "dynamic":
        gain_raw = g[1:] - g[:-1]
        valid_transition = valid_heads[:-1] & valid_heads[1:]

    elif mode == "fixed_source":
        
        w_A_fix = (anchor_mask[:-1].float() * c[1:]).unsqueeze(-1)
        w_O_fix = ((~anchor_mask[:-1]).float() * c[1:]).unsqueeze(-1)
        C_A_fix = w_A_fix.sum(dim=-2).squeeze(-1)
        C_O_fix = w_O_fix.sum(dim=-2).squeeze(-1)
        K_A_fix = (K[1:] * w_A_fix).sum(dim=-2) / C_A_fix.unsqueeze(-1).clamp_min(1e-8)
        K_O_fix = (K[1:] * w_O_fix).sum(dim=-2) / C_O_fix.unsqueeze(-1).clamp_min(1e-8)

        g_next_fix = torch.einsum("lhmd,lhd->lhm", Q[1:], K_A_fix - K_O_fix)
        gain_raw = g_next_fix - g[:-1]
        valid_fixed = (C_A_fix > 1e-8) & (C_O_fix > 1e-8)
        valid_transition = valid_heads[:-1] & valid_fixed

    else:
        raise ValueError(f"unsupport modes: {mode}")

    invalid = ~valid_transition.unsqueeze(-1)
    gain_raw = gain_raw.masked_fill(invalid, float("nan"))
    gain_scaled = gain_raw / math.sqrt(d_k)     

    scope_summaries = {}
    for scope_name, scope_mask in scopes.items():
        if scope_name == "is_mask_key":
            continue

        scope_mask = scope_mask.to(device=device, dtype=torch.bool)
        if scope_mask.ndim != 1 or scope_mask.numel() != M:
            raise ValueError(f"Scope {scope_name!r} must have shape [{M}].")

        raw_sub_gain = gain_raw[:, :, scope_mask]
        scaled_sub_gain = gain_scaled[:, :, scope_mask]

        head_gain_sum = torch.nansum(raw_sub_gain, dim=-1)
        head_gain_mean = torch.nanmean(raw_sub_gain, dim=-1)
        head_gain_sum[~valid_transition] = float("nan")
        head_gain_mean[~valid_transition] = float("nan")

        head_gain_sum_scaled = torch.nansum(scaled_sub_gain, dim=-1)
        head_gain_mean_scaled = torch.nanmean(scaled_sub_gain, dim=-1)
        head_gain_sum_scaled[~valid_transition] = float("nan")
        head_gain_mean_scaled[~valid_transition] = float("nan")

        valid_head_count = valid_transition.sum(dim=-1)
        layer_gain_sum = torch.nansum(head_gain_sum, dim=-1)
        layer_gain_mean = torch.nanmean(head_gain_mean, dim=-1)
        layer_gain_sum_scaled = torch.nansum(head_gain_sum_scaled, dim=-1)
        layer_gain_mean_scaled = torch.nanmean(head_gain_mean_scaled, dim=-1)

        no_value = valid_head_count == 0
        if not bool(scope_mask.any()):
            no_value = torch.ones_like(no_value)
            head_gain_sum.fill_(float("nan"))
            head_gain_mean.fill_(float("nan"))
            head_gain_sum_scaled.fill_(float("nan"))
            head_gain_mean_scaled.fill_(float("nan"))
        layer_gain_sum[no_value] = float("nan")
        layer_gain_mean[no_value] = float("nan")
        layer_gain_sum_scaled[no_value] = float("nan")
        layer_gain_mean_scaled[no_value] = float("nan")

        scope_summaries[scope_name] = {
            "head_gain_sum": head_gain_sum,
            "head_gain_mean": head_gain_mean,
            "layer_gain_sum": layer_gain_sum,
            "layer_gain_mean": layer_gain_mean,
            "head_gain_sum_scaled": head_gain_sum_scaled,
            "head_gain_mean_scaled": head_gain_mean_scaled,
            "layer_gain_sum_scaled": layer_gain_sum_scaled,
            "layer_gain_mean_scaled": layer_gain_mean_scaled,
            "valid_head_count": valid_head_count,
            "raw_sub_gain": raw_sub_gain,
            "scaled_sub_gain": scaled_sub_gain,
        }

    return {
        "scope_summaries": scope_summaries,
        "anchor_mask": anchor_mask,
        "C_A": C_A,
        "C_O": C_O,
        "K_A": K_A,
        "K_O": K_O,
        "valid_heads": valid_heads,
        "valid_transition": valid_transition,
    }
    
    
def analyze_single_layer(
    Q,
    K,
    A,
    is_mask_key=None,
    window_size=None,
    q_threshold=0.30,
    min_anchors=2,
    tail_fraction=0.25,
    tail_margin=0.10,
):
    Q = Q.float()
    K = K.float()
    A = A.float()

    device = Q.device
    
    #print(device)
    d_k = Q.shape[-1]

    if Q.shape[0] != K.shape[0]:
        K = K.repeat_interleave(
            Q.shape[0] // K.shape[0],
            dim=0,
        )

    if is_mask_key is not None:
        is_mask_key = is_mask_key.to(
            device=device,
            dtype=torch.bool,
        )

    
    ca_metrics = compute_shifted_c_gamma_CA(A)
    c = ca_metrics["c"]
    gamma = ca_metrics["gamma"]
    Error = ca_metrics["E"]
    
    A_frobenius_norm = torch.linalg.matrix_norm(
        A,
        ord="fro",
        dim=(-2, -1),
    )

    E_frobenius_norm = torch.linalg.matrix_norm(
        Error,
        ord="fro",
        dim=(-2, -1),
    )

    E_A_frobenius_ratio = (
        E_frobenius_norm
        / A_frobenius_norm.clamp_min(1e-12)
    )

    
    anchor_mask = detect_anchors(
        A,
        window_size=window_size,
        # sq_threshold=q_threshold,
        tail_fraction=tail_fraction,
        tail_margin=tail_margin,
        is_mask_key=is_mask_key,
    )

    
    (
        delta_K,
        C_A,
        C_O,
        K_A,
        K_O,
        valid_heads,
    ) = compute_prototypes(
        K,
        anchor_mask,
        c,
        min_anchors=min_anchors,
    )

    # g[l,h,i] = q[l,h,i]^T (K_A - K_O)
    g = torch.einsum(
        "hmd,hd->hm",
        Q,
        delta_K,
    )

    return {
        "c": c,
        "gamma": gamma,
        "anchor_mask": anchor_mask,
        "C_A": C_A,
        "C_O": C_O,
        "K_A": K_A,
        "K_O": K_O,
        "valid_heads": valid_heads,
        "g": g,
        "d_k": d_k,
        "E_A_frobenius_ratio": E_A_frobenius_ratio
    }


def analyze_adjacent_layers(
    source,
    target,
    scopes,
):
    """
    Compute dynamic transition gain between two adjacent layers.

    The transition gain is defined as the difference between the target
    and source layer gains. Invalid head transitions are excluded from
    the analysis. For each specified scope, the function computes
    per-head and layer-level gain sum
    
    """

    gain_raw = (
        target["g"]
        - source["g"]
    )

    valid_transition = (
        source["valid_heads"]
        & target["valid_heads"]
    )

    gain_raw = gain_raw.masked_fill(
        ~valid_transition.unsqueeze(-1),
        float("nan"),
    )

    gain_scaled = (
        gain_raw
        / math.sqrt(source["d_k"])
    )

    scope_summaries = {}

    for scope_name, scope_mask in scopes.items():
        if scope_name == "is_mask_key":
            continue

        scope_mask = scope_mask.cpu().bool()

        raw_sub_gain = gain_raw[:, scope_mask]
        scaled_sub_gain = gain_scaled[:, scope_mask]

        head_gain_sum = torch.nansum(
            raw_sub_gain,
            dim=-1,
        )
        head_gain_mean = torch.nanmean(
            raw_sub_gain,
            dim=-1,
        )

        head_gain_sum_scaled = torch.nansum(
            scaled_sub_gain,
            dim=-1,
        )
        head_gain_mean_scaled = torch.nanmean(
            scaled_sub_gain,
            dim=-1,
        )

        head_gain_sum[~valid_transition] = float("nan")
        head_gain_mean[~valid_transition] = float("nan")
        head_gain_sum_scaled[
            ~valid_transition
        ] = float("nan")
        head_gain_mean_scaled[
            ~valid_transition
        ] = float("nan")

        valid_head_count = valid_transition.sum()

        if valid_head_count == 0 or not scope_mask.any():
            layer_gain_sum = torch.tensor(float("nan"))
            layer_gain_mean = torch.tensor(float("nan"))
            layer_gain_sum_scaled = torch.tensor(float("nan"))
            layer_gain_mean_scaled = torch.tensor(float("nan"))
        else:
            layer_gain_sum = torch.nansum(
                head_gain_sum
            )
            layer_gain_mean = torch.nanmean(
                head_gain_mean
            )
            layer_gain_sum_scaled = torch.nansum(
                head_gain_sum_scaled
            )
            layer_gain_mean_scaled = torch.nanmean(
                head_gain_mean_scaled
            )

        scope_summaries[scope_name] = {
            "head_gain_sum": head_gain_sum,
            "head_gain_mean": head_gain_mean,
            "head_gain_sum_scaled": head_gain_sum_scaled,
            "head_gain_mean_scaled": head_gain_mean_scaled,
            "layer_gain_sum": layer_gain_sum,
            "layer_gain_mean": layer_gain_mean,
            "layer_gain_sum_scaled": layer_gain_sum_scaled,
            "layer_gain_mean_scaled": layer_gain_mean_scaled,
            "valid_head_count": valid_head_count,
        }

    return {
        "scope_summaries": scope_summaries,
        "valid_transition": valid_transition,
    }
    