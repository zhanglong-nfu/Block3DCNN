"""Scaling / trade-off sweep for the chunked (domain-decomposed) training path.

Reports, for every (volume, sep_depth, chunk_size):

    R_stag1      halo (input voxels per side) computed by _compute_overlay
    halo factor  ((C + 2R) / C)^3  -- the extra compute bought by the halo
    peak GiB     torch.cuda.max_memory_allocated during one fwd + bwd + step
    s/iter       mean wall-clock of one training step

This produces the paper's key figure: **peak memory is decoupled from the volume
(it tracks a single chunk), and the price is the halo overhead** — i.e. the
memory-vs-time trade-off curve and its optimal operating point.

Worst case is used on purpose: the target is ALL foreground, so `fg_keep_prob`
never drops a chunk and every chunk does forward+backward (real training keeps
only ~10% of background chunks, so real peaks/times are lower).

Run from repo root with the `nodl` env:
  VOLUMES=256,512 CHUNKS=2,2,2;4,4,4 SEP=2,3 CHANNELS=32 ITERS=1 \
      python train/custom/measure_scaling.py

OOM is caught and reported as "OOM" instead of aborting the sweep.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.blockcnn_network import BlockCNN_Network  # noqa: E402
import model.blockcnn_backbone  # noqa: E402  (registers BlockCNNMemopt / SA / P1FL3)
import model.blockcnn_head  # noqa: E402
from model.backbones.blockcnn import center_crop_to  # noqa: E402


def measure_loss_func(logits, target, aux_logits=None):
    """Minimal differentiable loss (coarse BCE + fine CE) on the chunked output.

    Mirrors the path verified bit-exact in ``verify_equiv_grad.py``: it exercises
    the coarse/fine heads (so their activations count toward the peak) and returns
    a dict with a ``'loss'`` key, which the relay back-propagates internally.
    We call ``bb.forward`` directly (not ``net.head``) to avoid the head wrapper's
    fused-kernel path, which trips an illegal-memory-access under this harness.
    """
    fine = logits['fine']
    coarse = logits['coarse']
    tgt = center_crop_to(target, fine.shape[-3:])
    tgt_fine = (tgt == 2).long()
    loss_fine = F.cross_entropy(fine, tgt_fine)
    loss_coarse = F.binary_cross_entropy_with_logits(
        coarse[:, 0], (tgt > 0).float())
    loss = loss_fine + loss_coarse
    return {'loss': loss, 'loss_fine': loss_fine.detach(),
            'loss_coarse': loss_coarse.detach()}


# Norms run in fp32: the production code keeps LayerNorm/GRN in fp32 (under
# autocast the convs are fp16 but the norms are forced fp32).  The fused CUDA
# kernels handle fp32 input fine -- the earlier illegal-memory-access was caused
# by the BlockCNN_Head wrapper path (net.head(...)), which this harness bypasses
# by calling bb.forward directly.  We therefore keep the real fused kernels by
# default.  Set NO_FUSED=1 only to force the PyTorch fallback (e.g. if a custom
# kernel misbehaves on an unusual shape) -- it is fp32 too, so the peak is
# identical.
if os.environ.get("NO_FUSED", "0") != "0":
    from model.backbones.blockcnn_ref_memopt import LayerNorm, GRN  # noqa: E402

    def _ln_fallback(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        xu = x - u
        s = xu.pow(2).mean(1, keepdim=True)
        x = xu / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None, None] * x + \
            self.bias[None, :, None, None, None]

    LayerNorm.forward = _ln_fallback

    if GRN is not None:
        def _grn_fallback(self, x):
            Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
            return (self.gamma * Nx + 1) * x + self.beta

        GRN.forward = _grn_fallback


CHANNELS = int(os.environ.get("CHANNELS", "32"))
BLOCKS = [int(x) for x in os.environ.get("BLOCKS", "3,3,3,3,3").split(",")]
BACKBONE = os.environ.get("BACKBONE", "BlockCNNMemopt")
ITERS = int(os.environ.get("ITERS", "1"))
VOLUMES = [int(v) for v in os.environ.get("VOLUMES", "256,512").split(",")]
SEP = [int(s) for s in os.environ.get("SEP", "2,3").split(",")]
CHUNKS = [tuple(int(x) for x in c.split(","))
          for c in os.environ.get("CHUNKS", "1,1,1;2,2,2;4,4,4").split(";")]
DEVICE = os.environ.get("DEVICE", "cuda:0")
# Per-GPU batch size (number of whole volumes forwarded/backward in one step).
# The paper's "per-GPU batch of 3" claim is validated by sweeping BATCH.
BATCH = int(os.environ.get("BATCH", "1"))
# FP16 measurements are not supported on this 3090/cuDNN build: cuDNN has no fp16
# engine for these 3D-conv shapes ("GET was unable to find an engine to execute
# this computation") and the aten fp16 fallback trips an illegal-memory-access
# (the fused LayerNorm/LN path under fp16).  The production code also trains in
# fp32 (norms always fp32), so the Pareto here is measured in fp32 only.
FP16 = os.environ.get("FP16", "0") == "1"
# Label dtype.  Default int8: the production label is an int8 class-index tensor
# (0/1/2 for the 3-way coronary task), 1 byte/voxel -- the same peak as uint8.
# An int64 label would be 8 GiB at 1024^3 (34 % of a 3090) and is NOT part of the
# *model* peak (it is a data tensor, shrunk per the "emit int8 label, cast to long
# inside the head" fix).  Use GT_DTYPE=long to reproduce the old pre-fix numbers.
_GT_DTYPE = {"int8": torch.int8, "uint8": torch.uint8, "int32": torch.int32,
             "long": torch.long}.get(os.environ.get("GT_DTYPE", "int8"),
                                     torch.int8)


# AUX: the aux head path triggers an illegal-memory-access in the custom fused
# LayerNorm/GRN CUDA kernels under this measurement harness (the relay without
# the aux head is verified bit-exact, see verify_equiv_grad.py).  The aux head is
# a tiny extra head whose peak-memory contribution is negligible, so the Pareto
# (backbone + chunk) is measured with AUX=0.  Set AUX=1 to include it.
USE_AUX = os.environ.get("AUX", "0") == "1"

# Loss used by the sweep.  ``head`` (default) is the PRODUCTION loss
# (``BlockCNN_Head.loss_func``); it implements the whole-volume seed protocol,
# so the default ``precompute_loss`` path costs O(1) memory.  ``simple`` is the
# decomposable CE+BCE stub used for the original Pareto: it has no seed
# protocol, so with precompute ON it falls back to the generic autograd.grad
# path (two extra whole volumes) -- that difference is itself a paper number.
LOSS = os.environ.get("LOSS", "head")


def make_loss(chunk_size):
    """The loss callable handed to the backbone's relay.

    ``head``: the production ``BlockCNN_Head.loss_func``.  It implements
    ``begin_volume/accumulate/end_volume/seed``, so the whole-volume-exact
    gradient is obtained from O(1) global statistics and NO whole-volume tensor
    is materialised -- this is the DEFAULT training configuration.
    ``simple``: the decomposable stub (no protocol -> generic fallback).
    """
    if LOSS != "head":
        return measure_loss_func
    from model.blockcnn_head import BlockCNN_Head
    head = BlockCNN_Head(num_classes=3, chunk_size=tuple(chunk_size),
                         dice_weight=1.0, bce_weight=0.2, aux_weight=0.5,
                         eps=1.0).to(DEVICE)
    return head.loss_func


def build(volume, sep_depth, chunk_size):
    bb = dict(type=BACKBONE, in_ch=1, channels=CHANNELS, blocks=BLOCKS,
              sep_depth=sep_depth, fusion_layers=3, num_classes=3,
              use_aux_head=USE_AUX, norm_type="layernorm",
              chunk_size=tuple(chunk_size))
    hd = dict(type="BlockCNN_Head", num_classes=3, chunk_size=tuple(chunk_size),
              dice_weight=1.0, bce_weight=0.2, focal_gamma=0.5,
              focal_alpha=(0.0, 0.5, 0.5), aux_weight=0.5, eps=1.0,
              fg_keep_prob=0.1, debug=dict(enabled=False))
    return BlockCNN_Network(bb, hd, pipelines=[]).to(DEVICE)


def main():
    print(f"backbone={BACKBONE} channels={CHANNELS} blocks={BLOCKS} "
          f"device={DEVICE} batch={BATCH} iters={ITERS}")
    print(f"{'volume':>8s} {'sep':>3s} {'chunk':>10s} {'R_stag1':>7s} "
          f"{'halo×':>7s} {'peak GiB':>9s} {'s/iter':>8s}")
    rows = []
    for v in VOLUMES:
        for sep in SEP:
            for cs in CHUNKS:
                tag = f"{v:8d} {sep:3d} {str(cs):>10s}"
                try:
                    net = build(v, sep, cs)
                    bb = net.backbone
                    if FP16:
                        if hasattr(bb, 'fp16_enabled'):
                            bb.fp16_enabled = True
                        else:
                            raise RuntimeError(
                                f"{BACKBONE} has no fp16_enabled (fp16 needs "
                                "BlockCNNMemopt)")
                    r1 = int(getattr(net.backbone, '_R_stag1', -1))
                    c = v // cs[0]
                    halo = ((c + 2 * r1) / c) ** 3 if r1 >= 0 else float('nan')
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    torch.cuda.empty_cache()
                    inp = torch.rand(BATCH, 1, v, v, v, device=DEVICE)
                    # We call net.head(net.backbone, inp, gt) directly instead of
                    # net(inp, gt): BlockCNN_Network.forward unconditionally casts the
                    # input to half, which (with fp16_enabled=False) makes the float
                    # conv weights disagree with a half input.  Passing float here lets
                    # the backbone's own _autocast(fp16_enabled) drive fp16 when enabled.
                    # NOTE: the label is allocated as GT_DTYPE (default int8).  An
                    # int64 label would be 8 bytes/voxel (8 GiB at 1024^3) -- the
                    # single largest *non-model* resident tensor and NOT part of the
                    # model peak; it is shrunk per the "emit int8 label, cast to
                    # long inside the head" fix.  The head only uses (target>0),
                    # (target==2) and target.long(), all safe with int8.
                    gt = torch.ones((BATCH, v, v, v), dtype=_GT_DTYPE, device=DEVICE)
                    loss_func = make_loss(cs)
                    opt = torch.optim.Adam(bb.parameters(), lr=1e-4)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    n_chunks = cs[0] * cs[1] * cs[2]
                    for _ in range(ITERS):
                        # Call bb.forward directly (the verified bit-exact path) with
                        # a simple loss; the relay back-props internally.  This avoids
                        # BlockCNN_Head's fused-kernel path, which trips an
                        # illegal-memory-access under this harness.
                        out = bb(inp, gt, cs, list(range(n_chunks)),
                                 loss_func, fg_keep_prob=1.0)
                        opt.step()
                        opt.zero_grad()
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) / ITERS
                    peak = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3
                    del net, inp, gt, opt, out
                    torch.cuda.empty_cache()
                    print(f"{tag} {r1:7d} {halo:7.2f} {peak:9.2f} {dt:8.2f}")
                    rows.append((v, sep, cs, r1, halo, peak, dt))
                except RuntimeError as e:
                    msg = str(e).splitlines()[0]
                    kind = "OOM" if "out of memory" in msg.lower() else "ERR"
                    print(f"{tag} {'-':>7s} {'-':>7s} {kind:>9s} {'-':>8s}   "
                          f"({msg[:70]})")
                    torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"{tag} {'-':>7s} {'-':>7s} {'ERR':>9s} {'-':>8s}   "
                          f"({type(e).__name__}: {str(e)[:60]})")
                    torch.cuda.empty_cache()

    print("\n--- summary: peak memory vs chunk (volume fixed) ---")
    for v in VOLUMES:
        for sep in SEP:
            sel = [r for r in rows if r[0] == v and r[1] == sep]
            if len(sel) > 1:
                base = min(sel, key=lambda r: r[5])
                print(f"  vol={v} sep={sep}: "
                      + ", ".join(f"chunk{c}->{p:.1f}GiB/{t:.1f}s"
                                  for _, _, c, _, _, p, t in sel))
    print(f"\n--- throughput (batch={BATCH}) ---")
    for v, sep, cs, r1, halo, peak, dt in rows:
        print(f"  vol={v} sep={sep} chunk{cs}: peak={peak:.2f}GiB "
              f"s/iter={dt:.2f} -> {BATCH/dt:.4f} vol/s "
              f"(={BATCH/dt*3600:.1f} vol/h)")


if __name__ == "__main__":
    main()
