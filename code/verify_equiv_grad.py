"""Training-time GRADIENT equivalence: chunked grad-relay BlockCNN vs an
independent whole-volume reference network.

Why this exists
---------------
The paper's central claim is that the chunked (domain-decomposed) training path
is not an approximation: it only re-orders the computation and accumulates the
per-chunk gradients, so the loss and every parameter gradient must equal those
of an equivalent whole-volume network trained in one forward/backward.

  * Inference-side exact tiling is the NIST-style argument
    (Majurski & Bajcsy, NIST J. Res. 126, 2021) -- halo restores the receptive
    field so each output voxel sees the same data as in the whole-volume pass.
  * Training-side exactness follows because gradient ACCUMULATION over chunks
    is a sum of the same per-voxel contributions: the chunked path computes
    dL/dtheta = sum_c dL_c/dtheta, which is exactly what one whole-volume
    backward computes.  Only the summation order changes.

This harness verifies that claim numerically:

    reference  : a rebuilt STANDARD UNet (whole volume, padding=1 everywhere,
                 one graph, one backward) sharing BlockCNN's weights.
    candidates : the real relay path at chunk_size (1,1,1) / (2,2,2) / (4,4,4).

and compares, per parameter: loss, forward logits and max|dgrad|.

NOTE: the pre-existing ``verify_equiv.py`` is stale (its reference still uses
ConvTranspose + a single ``head``, while the backbone now uses nn.Upsample +
coarse/fine heads).  This script is written against the CURRENT topology and
uses the real ``BlockCNN`` (aux head off), not the stale BlockCNNVerify copy.

Run (smoke):
    cd train/custom
    DEVICE=cuda:0 VOLUME=256,256,256 CHUNK=2,2,2 python verify_equiv_grad.py
"""

import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.backbones.blockcnn import (  # noqa: E402
    BlockCNN, DoubleConv, make_res_layer, BinaryHead, MultiscalFusion,
    center_crop_to, NORM_TYPES,
)
# blockcnn_ref_memopt.py is a self-contained copy whose class is also named
# ``BlockCNN`` (the production memory-optimised relay + gradient relay).
from model.backbones.blockcnn_ref_memopt import (  # noqa: E402
    BlockCNN as BlockCNNMemopt,
)
import model.backbones.blockcnn_p1fl3 as p1  # noqa: E402
from model.backbones.blockcnn_fl3 import (  # noqa: E402
    BlockCNNFL3,
)

# Which backbone to verify.  Default = the production relay (BlockCNNMemopt),
# base of BlockCNNSA / the currently configured model.
BACKBONE = os.environ.get("BACKBONE", "memopt")


# --------------------------------------------------------------------------- #
# Config (mirrors imagecas_8ch_config.py; env-overridable)
# --------------------------------------------------------------------------- #
def _load_cfg():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(here, "..", "config",
                                        "imagecas_8ch_config.py"))
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("imagecas_8ch_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CFG = _load_cfg()
_g = lambda name, dflt: getattr(_CFG, name, dflt) if _CFG is not None else dflt  # noqa: E731

IN_CH = int(_g("in_ch", 1))
# channels: config default, env-overridable.  NOTE P1FL3's BinaryHead uses
# in_ch // 4 hidden dims, so channels must be >= 4 (use 8 for P1FL3).
CHANNELS = int(os.environ.get("CHANNELS", _g("channels", 2)))
BLOCKS = list(_g("blocks", [3, 3, 3, 3, 3]))
SEP_DEPTH = int(_g("sep_depth", 2))
FUSION_LAYERS = int(_g("fusion_layers", 3))
NUM_CLASSES = int(_g("num_classes", 3))
NORM_TYPE = os.environ.get("NORM_TYPE", str(_g("norm_type", "layernorm")))
VOLUME = [int(x) for x in os.environ.get("VOLUME", "256,256,256").split(",")]
CHUNK = tuple(int(v) for v in os.environ.get("CHUNK", "2,2,2").split(","))
DEVICE = os.environ.get("DEVICE", str(_g("device", "cuda:0")))
DTYPE = torch.float64 if os.environ.get("DTYPE") == "64" else torch.float32
TOL = float(os.environ.get("TOL", "1e-5"))
FG_KEEP_PROB = 1.0        # force every chunk -> deterministic vs reference
PADDED_REF = os.environ.get("PADDED_REF", "0") == "1"
CHECKPOINT_REF = os.environ.get("CHECKPOINT_REF", "0") == "1"  # fit 512+/1024³ ref

assert NORM_TYPE in NORM_TYPES, f"unknown NORM_TYPE={NORM_TYPE}"


# --------------------------------------------------------------------------- #
# Verification loss: the SAME differentiable function on both sides
# --------------------------------------------------------------------------- #
def verify_loss_func(logits, target, aux_logits=None):
    """Plain CE on the fine head + BCE-with-logits on the coarse head.

    Deliberately simple (no Dice / Focal / sampling) so that any difference we
    measure comes from the chunked execution, not from the loss.  Returns a
    dict with key ``loss``, matching the backbone's expected protocol.
    """
    fine = logits['fine']
    coarse = logits['coarse']
    tgt = center_crop_to(target, fine.shape[-3:])
    # NOTE: the fine head is hard-wired to 2 logits (fg1 vs fg2), so the
    # verification target is a deterministic 2-class projection of the 3-class
    # GT (fg2 vs rest).  Any deterministic function works: we only test whether
    # the chunked path reproduces the whole-volume gradient.
    tgt_fine = (tgt == 2).long()
    loss_fine = F.cross_entropy(fine, tgt_fine)
    loss_coarse = F.binary_cross_entropy_with_logits(
        coarse[:, 0], (tgt > 0).float())
    return {'loss': loss_fine + loss_coarse,
            'loss_fine': loss_fine.detach(),
            'loss_coarse': loss_coarse.detach()}


def verify_dice_loss_func(logits, target, aux_logits=None):
    """Soft Dice loss -- the Dice term of the real training objective -- on BOTH
    the fine (3-class) and coarse (binary) heads, computed IDENTICALLY on the
    candidate relay and the whole-volume reference.  Drop-in for
    ``verify_loss_func`` (returns ``{'loss': ...}``).

    Why a separate loss: Table~1's ``loss`` / ``grad`` rows were measured with
    the simple verification CE+BCE; they must also hold for the Dice loss the
    network is actually trained with.  The relay's ``_precompute_whole_volume_loss``
    differentiates the ASSEMBLED whole-volume prediction, so the Dice term
    (a ratio of whole-volume sums) is evaluated and differentiated exactly -- the
    chunked path reproduces the whole-volume Dice loss and every parameter
    gradient bit-for-bit.
    """
    fine = logits['fine']
    coarse = logits['coarse']
    tgt = center_crop_to(target, fine.shape[-3:])
    eps = 1e-6
    # fine: soft multi-class Dice over foreground classes (1=LCA, 2=RCA);
    # background (0) is excluded from the mean, matching training.
    p = F.softmax(fine, dim=1)
    fg_dices = []
    for c in (1, 2):
        pc = p[:, c]
        gc = (tgt == c).float()
        inter = (pc * gc).sum()
        denom = pc.sum() + gc.sum() + eps
        fg_dices.append(2.0 * inter / denom)
    fine_dice = torch.stack(fg_dices).mean()
    # coarse: binary soft Dice (vessel = GT > 0).
    pc = torch.sigmoid(coarse[:, 0])
    gc = (tgt > 0).float()
    inter = (pc * gc).sum()
    denom = pc.sum() + gc.sum() + eps
    coarse_dice = 2.0 * inter / denom
    dice = (fine_dice + coarse_dice) / 2.0
    return {'loss': 1.0 - dice,
            'loss_fine': fine_dice.detach(),
            'loss_coarse': coarse_dice.detach()}


class SeedVerifyLossFunc:
    """Same loss as :func:`verify_loss_func` (fine CE + coarse BCE-with-logits)
    but implementing the O(1) seed protocol (``begin_volume`` / ``accumulate``
    / ``end_volume`` / ``seed``) so the candidate relay takes the PRODUCTION
    seed path (Branch 2) instead of the generic autograd.grad path (Branch 1).

    The underlying per-voxel loss is identical to ``verify_loss_func``, so the
    whole-volume reference (which always uses ``verify_loss_func``) remains the
    correct ground truth.
    """

    def __call__(self, logits, target, aux_logits=None):
        # Plain callable used by the whole-volume reference.
        fine = logits['fine']
        coarse = logits['coarse']
        tgt = center_crop_to(target, fine.shape[-3:])
        tgt_fine = (tgt == 2).long()
        loss_fine = F.cross_entropy(fine, tgt_fine)
        loss_coarse = F.binary_cross_entropy_with_logits(
            coarse[:, 0], (tgt > 0).float())
        return {'loss': loss_fine + loss_coarse,
                'loss_fine': loss_fine.detach(),
                'loss_coarse': loss_coarse.detach()}

    # -- O(1) seed protocol ------------------------------------------------- #
    def begin_volume(self):
        self._sum_fine = 0.0
        self._sum_coarse = 0.0
        self._count = 0

    def accumulate(self, logits, target, aux):
        fine = logits['fine']
        coarse = logits['coarse']
        tgt = center_crop_to(target, fine.shape[-3:])
        tgt_fine = (tgt == 2).long()
        vox = fine.shape[-3] * fine.shape[-2] * fine.shape[-1]
        self._sum_fine += float(
            F.cross_entropy(fine, tgt_fine, reduction='sum'))
        self._sum_coarse += float(F.binary_cross_entropy_with_logits(
            coarse[:, 0], (tgt > 0).float(), reduction='sum'))
        self._count += vox * fine.shape[0]

    def end_volume(self):
        # The whole-volume mean loss (the value reported as ``loss``).
        loss = (self._sum_fine + self._sum_coarse) / max(self._count, 1)
        return {'loss': loss,
                'loss_fine': self._sum_fine / max(self._count, 1),
                'loss_coarse': self._sum_coarse / max(self._count, 1)}

    def seed(self, logits, target, aux):
        # dL_whole / d(logits_b) for THIS block, where L_whole = (1/N) * sum of
        # per-voxel losses and N is the global voxel count (set in end_volume).
        # Embedding the (1/N) factor in the seed makes the relay's contraction
        # <y_b, seed_b> have gradient exactly dL_whole/dtheta.
        fine = logits['fine']
        coarse = logits['coarse']
        tgt = center_crop_to(target, fine.shape[-3:])
        tgt_fine = (tgt == 2).long()
        p = F.softmax(fine, dim=1)
        g_fine = p.clone()
        g_fine.scatter_(1, tgt_fine.unsqueeze(1), g_fine.gather(
            1, tgt_fine.unsqueeze(1)) - 1.0)
        s_coarse = torch.sigmoid(coarse[:, 0])
        g_coarse = (s_coarse - (tgt > 0).float()).unsqueeze(1)
        inv_n = 1.0 / max(self._count, 1)
        return {'coarse': g_coarse * inv_n, 'fine': g_fine * inv_n}

# --------------------------------------------------------------------------- #
# Independent whole-volume reference (the ground truth)
# --------------------------------------------------------------------------- #
class StandardUNet(nn.Module):
    """Plain whole-volume UNet with BlockCNN's CURRENT topology.

    Same low-level blocks (DoubleConv / make_res_layer / BinaryHead /
    MultiscalFusion) but wired straight through: no chunking, no halo, no
    relay, and padding=1 everywhere -- which is the whole-volume counterpart of
    the chunked padding=0 + halo rule.
    """

    def __init__(self, in_ch, channels, blocks, sep_depth, fusion_layers,
                 num_classes, norm_type):
        super().__init__()
        depth = len(blocks) + 1
        channels = [channels * (2 ** min(d, 3)) for d in range(depth)]

        # Whole-volume reference: padding=1 everywhere (see padding policy).
        p = 1
        down_layers = [DoubleConv(in_ch, channels[0], stride=2, kernel_size=3,
                                  padding=p, norm_type=norm_type)]
        for idx in range(1, depth):
            down_layers.append(
                make_res_layer(channels[idx - 1], channels[idx],
                               blocks[idx - 1], stride=2, padding=p,
                               norm_type=norm_type))
        self.down_layers = nn.ModuleList(down_layers)

        self._n_up_s2 = depth - sep_depth - 1
        up_layers_s2, up_layers_s3 = [], []
        for pos, idx in enumerate(range(depth - 1, 0, -1)):
            dc = DoubleConv(channels[idx] + channels[idx - 1],
                            channels[idx - 1], padding=p,
                            norm_type=norm_type)
            (up_layers_s2 if pos < self._n_up_s2 else up_layers_s3).append(dc)
        self.up_layers_s2 = nn.ModuleList(up_layers_s2)
        self.up_layers_s3 = nn.ModuleList(up_layers_s3)

        self.up_list = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode='nearest')
            for _ in range(depth - 1, 0, -1))

        self._depth = depth
        self._sep_depth = sep_depth

        fl = min(fusion_layers, sep_depth)
        if fl > 1:
            in_chs = [channels[i] for i in range(fl)]
            scales = [2 ** i for i in range(fl)]
            self.fusion = MultiscalFusion(in_chs, channels[0], scales,
                                          norm_type=norm_type)
        else:
            self.fusion = None

        self.coarse_head = BinaryHead(channels[0], 1, norm_type=norm_type)
        # The fine head is hard-wired to 2 logits (fg1 vs fg2) in BlockCNN.
        self.fine_head = BinaryHead(channels[0], 3, norm_type=norm_type)

    def set_norm_expected(self, size):
        """Full-volume expected region (no-op for LayerNorm)."""
        for m in self.modules():
            if hasattr(m, 'set_expected'):
                m.set_expected(size)

    def forward(self, inp):
        s1_ints = []
        x = inp
        for layer in self.down_layers[:self._sep_depth]:
            x = layer(x)
            s1_ints.append(x)

        x_up = [x]
        for layer in self.down_layers[self._sep_depth:]:
            x = layer(x)
            x_up.append(x)
        x_up = x_up[::-1]

        x = x_up[0]
        for idx in range(self._n_up_s2):
            x = self.up_list[idx](x)
            x = torch.cat([x, x_up[idx + 1]], dim=1)
            x = self.up_layers_s2[idx](x)

        outputs = []
        x_up_s1 = s1_ints[::-1]
        for k, layer in enumerate(self.up_layers_s3):
            x = self.up_list[self._n_up_s2 + k](x)
            skip = center_crop_to(x_up_s1[k], x.shape[-3:])
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
            outputs.append(x)

        if self.fusion is not None:
            fusion_inputs = outputs[-min(self._sep_depth, len(outputs)):][::-1]
            x = self.fusion(fusion_inputs)

        return {'coarse': self.coarse_head(x), 'fine': self.fine_head(x)}


class StandardUNetFL3(nn.Module):
    """Whole-volume reference for **BlockCNN_P1FL3** (padding=1 everywhere).

    IMPORTANT (the P1FL3 vs plain-UNet difference): a standard UNet fuses only
    ``min(fusion_layers, sep_depth)`` branches, i.e. **2** for sep_depth=2.
    P1FL3 fuses **3**: the two s3 decoder outputs (scales 1 and 2) PLUS the
    cropped stag2 (trunk) output as the scale-4 branch ``b2``.  A plain
    2-branch reference is therefore NOT the equivalent network -- this class
    reproduces the 3-branch FL3 fusion on the whole volume, where ``b2`` is
    simply the stag2 output (no halo exists in a whole-volume pass).
    """

    def __init__(self, in_ch, channels, blocks, sep_depth, fusion_layers,
                 num_classes, norm_type=None):
        super().__init__()
        depth = len(blocks) + 1
        channels = [channels * (2 ** min(d, 3)) for d in range(depth)]

        down_layers = [p1.DoubleConv(in_ch, channels[0], stride=2,
                                     kernel_size=3)]
        for idx in range(1, depth):
            down_layers.append(
                p1.make_res_layer(channels[idx - 1], channels[idx],
                                  blocks[idx - 1], stride=2))
        self.down_layers = nn.ModuleList(down_layers)

        up_layers = []
        for idx in range(depth - 1, 0, -1):
            up_layers.append(
                p1.DoubleConv(channels[idx] + channels[idx - 1],
                              channels[idx - 1]))
        self.up_layers = nn.ModuleList(up_layers)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self._depth = depth
        self._sep_depth = sep_depth

        n_fuse = sep_depth + 1                       # FL3: sep_depth + 1 = 3
        in_chs = [channels[i] for i in range(n_fuse)]
        scales = [2 ** i for i in range(n_fuse)]
        self.fusion = p1.MultiscalFusion(in_chs, channels[0], scales)
        self._fusion_layers = n_fuse

        self.coarse_head = p1.BinaryHead(channels[0], 1)
        self.fine_head = p1.BinaryHead(channels[0], 3)

        # PADDED_REF mode: the chunked path folds (centre-crops) the s1 output
        # by ``_pad_stag1`` BEFORE feeding the whole-volume s2 trunk.  A
        # zero-padded whole-volume reference must do the same, otherwise s2 runs
        # on the halo-carrying s1 output and the two disagree everywhere.
        self.fold1 = 0
        # The s3 decoder consumes the stag2 feature surrounded by ``_R_stag3``
        # zeros (that is exactly the chunked path's ``stag2c`` halo).  A
        # reference that feeds the decoder the BARE stag2 feature instead only
        # sees each conv's own 1-voxel padding, which leaves a spurious volume
        # border band (this was the "rim" mistake).  Set from ``bb._R_stag3``.
        self.f2halo = 0

    @staticmethod
    def _match(x, skip):
        """Centre-crop ``x`` and ``skip`` to their COMMON spatial size.

        With halo-padded (non power-of-two) inputs the down path and the
        upsampled path can differ by one voxel (floor vs ceil at stride-2
        convs); both sides are cropped to the minimum, mirroring what the
        chunked path's ``_crop_s1_skip`` does for the s3 skips.
        """
        s = [min(x.shape[-3 + i], skip.shape[-3 + i]) for i in range(3)]
        return center_crop_to(x, s), center_crop_to(skip, s)

    def set_norm_expected(self, size):
        for m in self.modules():
            if hasattr(m, 'set_expected'):
                m.set_expected(size)

    def forward(self, inp):
        s1_ints = []
        x = inp
        for layer in self.down_layers[:self._sep_depth]:
            x = layer(x)
            s1_ints.append(x)

        if self.fold1:                      # fold the s1 halo away (see above)
            d = [s - 2 * self.fold1 for s in x.shape[-3:]]
            x = center_crop_to(x, d)

        x_up = [x]
        for layer in self.down_layers[self._sep_depth:]:
            x = layer(x)
            x_up.append(x)
        x_up = x_up[::-1]

        x = x_up[0]
        for idx in range(self._depth - self._sep_depth - 1):
            x = self.up(x)
            x, skip = self._match(x, x_up[idx + 1])
            x = torch.cat([x, skip], dim=1)
            x = self.up_layers[idx](x)

        F2 = x                                    # stag2 (trunk) output == b2
        if self.f2halo:
            # surround the stag2 feature by the same R3 zeros the chunked path
            # extracts (``stag2c``), so the decoder sees an identical boundary
            x = torch.nn.functional.pad(F2, (self.f2halo,) * 6)
        outputs = []
        x_up_s1 = s1_ints[::-1]
        for skip_idx, layer in enumerate(
                self.up_layers[self._depth - self._sep_depth - 1:]):
            x = self.up(x)
            x, skip = self._match(x, x_up_s1[skip_idx])
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
            outputs.append(x)

        x = self.fusion(outputs[::-1] + [F2])
        return {'coarse': self.coarse_head(x), 'fine': self.fine_head(x)}


class StandardUNetFL3Base(nn.Module):
    """Whole-volume reference for **BlockCNNFL3** (base, nearest up-sample).

    FL3 = base BlockCNN (padding=0 in the chunked s1/s3, padding=1 in the
    whole-volume s2, nearest up-sample) PLUS a 3-way fusion whose coarsest
    branch is the stag2 (trunk) output.  Its whole-volume equivalent therefore
    uses padding=1 everywhere, nearest up-sample, and the SAME 3-branch fusion
    -- this class.

    The module layout mirrors ``StandardUNet`` EXACTLY (same ``down_layers`` /
    ``up_layers_s2`` / ``up_layers_s3`` / ``up_list`` / ``fusion`` / head
    naming) so the parameter keys line up with ``BlockCNNFL3``; only the
    fusion becomes ``sep_depth + 1`` branches and the forward captures the
    stag2 trunk output ``F2`` as the coarsest branch.  (The P1 (padding=1
    everywhere) variant is ``StandardUNetFL3``.)
    """

    def __init__(self, in_ch, channels, blocks, sep_depth, fusion_layers,
                 num_classes, norm_type):
        super().__init__()
        depth = len(blocks) + 1
        channels = [channels * (2 ** min(d, 3)) for d in range(depth)]

        p = 1
        down_layers = [DoubleConv(in_ch, channels[0], stride=2, kernel_size=3,
                                  padding=p, norm_type=norm_type)]
        for idx in range(1, depth):
            down_layers.append(
                make_res_layer(channels[idx - 1], channels[idx],
                               blocks[idx - 1], stride=2, padding=p,
                               norm_type=norm_type))
        self.down_layers = nn.ModuleList(down_layers)

        self._n_up_s2 = depth - sep_depth - 1
        up_layers_s2, up_layers_s3 = [], []
        for pos, idx in enumerate(range(depth - 1, 0, -1)):
            dc = DoubleConv(channels[idx] + channels[idx - 1],
                            channels[idx - 1], padding=p,
                            norm_type=norm_type)
            (up_layers_s2 if pos < self._n_up_s2 else up_layers_s3).append(dc)
        self.up_layers_s2 = nn.ModuleList(up_layers_s2)
        self.up_layers_s3 = nn.ModuleList(up_layers_s3)

        self.up_list = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode='nearest')
            for _ in range(depth - 1, 0, -1))

        self._depth = depth
        self._sep_depth = sep_depth

        # FL3: fuse sep_depth + 1 branches (the two s3 outputs + the stag2 trunk).
        n_fuse = sep_depth + 1
        in_chs = [channels[i] for i in range(n_fuse)]
        scales = [2 ** i for i in range(n_fuse)]
        self.fusion = MultiscalFusion(in_chs, channels[0], scales,
                                      norm_type=norm_type)
        self._fusion_layers = n_fuse

        self.coarse_head = BinaryHead(channels[0], 1, norm_type=norm_type)
        self.fine_head = BinaryHead(channels[0], 3, norm_type=norm_type)

    def set_norm_expected(self, size):
        for m in self.modules():
            if hasattr(m, 'set_expected'):
                m.set_expected(size)

    def forward(self, inp):
        s1_ints = []
        x = inp
        for layer in self.down_layers[:self._sep_depth]:
            x = layer(x)
            s1_ints.append(x)

        x_up = [x]
        for layer in self.down_layers[self._sep_depth:]:
            x = layer(x)
            x_up.append(x)
        x_up = x_up[::-1]

        x = x_up[0]
        for idx in range(self._n_up_s2):
            x = self.up_list[idx](x)
            x = torch.cat([x, x_up[idx + 1]], dim=1)
            x = self.up_layers_s2[idx](x)

        F2 = x                                    # stag2 (trunk) output == b2
        outputs = []
        x_up_s1 = s1_ints[::-1]
        for k, layer in enumerate(self.up_layers_s3):
            x = self.up_list[self._n_up_s2 + k](x)
            skip = center_crop_to(x_up_s1[k], x.shape[-3:])
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
            outputs.append(x)

        x = self.fusion(outputs[::-1] + [F2])
        return {'coarse': self.coarse_head(x), 'fine': self.fine_head(x)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_inputs():
    torch.manual_seed(0)
    random.seed(0)
    inp = torch.randn(1, IN_CH, *VOLUME, dtype=DTYPE, device=DEVICE)
    target = torch.randint(0, NUM_CLASSES, (1, *VOLUME),
                           dtype=torch.long, device=DEVICE)
    return inp, target


def rel_diff(a: Tensor, b: Tensor) -> float:
    """max|a-b| normalised by mean|a| (scale-free)."""
    return ((a - b).abs().max() / (a.abs().mean() + 1e-8)).item()


def zero_grads(m):
    for p in m.parameters():
        p.grad = None


def sync_weights(ref: StandardUNet, bb: BlockCNN):
    sd_r, sd_b = ref.state_dict(), bb.state_dict()
    missing = set(sd_r) - set(sd_b)
    extra = set(sd_b) - set(sd_r)
    assert not missing and not extra, (
        f"param key mismatch:\n missing in backbone: {sorted(missing)}\n"
        f" extra in backbone: {sorted(extra)}")
    bb.load_state_dict({k: v.clone() for k, v in sd_r.items()})


def compare_grads(name, bb, ref, failures, tol=TOL):
    ref_state = dict(ref.named_parameters())
    worst, worst_key = 0.0, ""
    for key, p in bb.named_parameters():
        rp = ref_state.get(key)
        if p.grad is None and (rp is None or rp.grad is None):
            # Dead parameter (never used on EITHER side, e.g. the unused
            # per-branch fusion norms) -> nothing to compare, not a failure.
            print(f"  [{name}] grad {key}: dead on both sides (unused) -> skip")
            continue
        if rp is None or p.grad is None or rp.grad is None:
            print(f"  [{name}] grad {key}: MISSING (bb={p.grad is not None}, "
                  f"ref={None if rp is None else rp.grad is not None})")
            failures.append(f"  [{name}] grad {key}: missing")
            continue
        d = rel_diff(p.grad.detach(), rp.grad.detach())
        if d > worst:
            worst, worst_key = d, key
        if d > tol:
            failures.append(f"  [{name}] grad {key}: max|Δ|/mean = {d:.3e}")
    print(f"  [{name}] WORST grad diff = {worst:.3e} @ {worst_key} "
          f"(tol={tol:.1e})")
    return worst


def forward_test_full(bb, inp, chunk_size):
    """Run the chunked inference path and reassemble the full volume.

    ``forward_test`` is a generator that yields one chunk at a time (it must not
    materialise the whole padded/unfolded tensor, see the 2026-08-26 OOM fix).
    """
    import einops
    chunks, keys = {}, None
    with torch.no_grad():
        for out in bb.forward_test(inp.clone(), chunk_size):
            if keys is None:
                keys = list(out.keys())
            for k in keys:
                chunks.setdefault(k, []).append(out[k])
    d1, h1, w1 = chunk_size
    full = {}
    for k in keys:
        stacked = torch.cat(chunks[k], dim=0)
        full[k] = einops.rearrange(
            stacked, "(b d1 h1 w1) c d h w -> b c (d1 d) (h1 h) (w1 w)",
            d1=d1, h1=h1, w1=w1)
    return full


def run_case(bb, ref, inp, target, chunk_size, failures, cand_loss_func=None,
             loss_fn=None):
    # cand_loss_func: the loss fed to the candidate relay.  Defaults to the plain
    # ``verify_loss_func``.  Pass a seed-protocol loss (Branch 2) to exercise the
    # production O(1) seed path; the underlying loss must equal ``verify_loss_func``
    # so the whole-volume reference (which always uses ``verify_loss_func``) stays valid.
    # Resolve the loss used on BOTH sides (reference + candidate relay).  Defaults
    # to the plain verification CE+BCE so existing behaviour is unchanged.
    loss_fn = loss_fn or cand_loss_func or verify_loss_func
    cand_loss_func = cand_loss_func or loss_fn

    n_chunks = chunk_size[0] * chunk_size[1] * chunk_size[2]
    name = f"chunk={tuple(chunk_size)}"
    print(f"\n=== {name} : relay vs whole-volume reference ===")

    # --- reference: one whole-volume fwd/bwd -------------------------------
    # PADDED_REF: for padding=1 ("P1") backbones the chunked path pads the
    # volume with R1 zeros (halo) before running the SAME padding=1 convs, so
    # its boundary condition is "volume surrounded by R1 zeros", NOT "each conv
    # pads with 1 zero".  With PADDED_REF=1 the reference runs on the zero-padded
    # volume too and is centre-cropped back -- that is the whole-volume network
    # the P1 chunked path is actually equivalent to.
    zero_grads(ref)
    ref.set_norm_expected(tuple(VOLUME))
    if CHECKPOINT_REF:
        # Wrap the whole-volume reference forward in gradient checkpointing so the
        # reference fits 512^3+ on a 24 GB GPU (the unchunked baseline OOMs there).
        # Checkpointing is numerically EXACT, so equivalence numbers are unchanged.
        def _ref_fwd(r, x):
            return r(x)
    if PADDED_REF and hasattr(bb, '_R_stag1'):
        r1 = int(bb._R_stag1)
        ref.fold1 = int(getattr(bb, '_pad_stag1', 0))
        ref_inp = torch.nn.functional.pad(inp, (r1,) * 6)
        if CHECKPOINT_REF:
            ref_out = {k: center_crop_to(v, tuple(VOLUME))
                       for k, v in torch.utils.checkpoint.checkpoint(
                           _ref_fwd, ref, ref_inp, use_reentrant=False).items()}
        else:
            ref_out = {k: center_crop_to(v, tuple(VOLUME))
                       for k, v in ref(ref_inp).items()}
        print(f"  [{name}] reference runs on zero-padded volume "
              f"(pad={r1}) and is centre-cropped to {tuple(VOLUME)}")
    else:
        if CHECKPOINT_REF:
            ref_out = torch.utils.checkpoint.checkpoint(
                _ref_fwd, ref, inp, use_reentrant=False)
        else:
            ref_out = ref(inp)
    ref_loss = loss_fn(ref_out, target)['loss']
    ref_loss.backward()

    # --- candidate: real chunked relay path --------------------------------
    sync_weights(ref, bb)
    zero_grads(bb)
    bb.eval() if False else None
    report = bb.forward(inp, target, chunk_size, list(range(n_chunks)),
                        cand_loss_func, fg_keep_prob=FG_KEEP_PROB)
    assert report is not None, "relay selected no chunk"
    bb_loss = report['loss']

    # Emulate the training-time trainer unscale of the AMP loss-scaling factor
    # (``BlockCNNMemopt._grad_scale``, default 1e5).  The relay multiplies every
    # per-chunk backward scalar by this factor (an fp16 seed underflow guard);
    # the trainer divides the parameter grads back by it before clip/step, so the
    # *effective* parameter update is exact and equals the whole-volume reference.
    # The verification has no trainer, so we apply the identical unscale here:
    # without it the candidate grads are ``_grad_scale`` x too large and the
    # comparison fails -- that is a harness artefact, not a real discrepancy.
    gscale = float(getattr(bb, '_grad_scale', 1.0) or 1.0)
    if gscale != 1.0:
        # The relay multiplies every per-chunk backward scalar by ``gscale``; the
        # trainer divides the parameter grads back by it before clip/step, so the
        # effective update is exact.  Emulate that unscale here (gradients only --
        # ``report['loss']`` is already the true, unscaled whole-volume loss).
        for p in bb.parameters():
            if p.grad is not None:
                p.grad.div_(gscale)

    # --- forward logits (chunked whole-volume pass of the same weights) ----
    bb_out = forward_test_full(bb, inp, chunk_size)
    for k in ('coarse', 'fine'):
        a, b = ref_out[k], bb_out[k]
        s = [min(a.shape[-3 + i], b.shape[-3 + i]) for i in range(3)]
        if s != list(a.shape[-3:]) or s != list(b.shape[-3:]):
            print(f"  [note] {k}: ref {tuple(a.shape[-3:])} vs bb "
                  f"{tuple(b.shape[-3:])} -> compare on common {tuple(s)}")
        pa = [(a.shape[-3 + i] - s[i]) // 2 for i in range(3)]
        pb = [(b.shape[-3 + i] - s[i]) // 2 for i in range(3)]
        ca = a[:, :, pa[0]:pa[0] + s[0], pa[1]:pa[1] + s[1], pa[2]:pa[2] + s[2]]
        cb = b[:, :, pb[0]:pb[0] + s[0], pb[1]:pb[1] + s[1], pb[2]:pb[2] + s[2]]
        d = rel_diff(ca, cb)
        status = "OK" if d <= TOL else "FAIL"
        print(f"  [{name}] {k} logits: max|Δ|/mean = {d:.3e}  {status}")
        if d > TOL:
            failures.append(f"  [{name}] {k} logits: {d:.3e}")
        # --- border-rim scan: is the error confined to a rim? ----------------
        adm = (ca - cb).abs()
        scan = []
        for rim in (0, 2, 4, 8, 12, 16, 24, 32):
            if min(s) <= 2 * rim + 1:
                break
            inner = adm if rim == 0 else \
                adm[..., rim:-rim, rim:-rim, rim:-rim]
            scan.append(f"rim{rim}={inner.max().item():.2e}")
        print(f"  [{name}] {k} rim scan (abs): " + "  ".join(scan))

    d_loss = abs(float(bb_loss) - float(ref_loss)) / (abs(float(ref_loss)) + 1e-8)
    status = "OK" if d_loss <= TOL else "FAIL"
    print(f"  [{name}] loss: ref={float(ref_loss):.8f} bb={float(bb_loss):.8f} "
          f"|Δ|/|ref| = {d_loss:.3e}  {status}")
    if d_loss > TOL:
        failures.append(f"  [{name}] loss: {d_loss:.3e}")

    compare_grads(name, bb, ref, failures)
    ref.set_norm_expected(None)
    return d_loss


def main():
    print(f"device={DEVICE} dtype={DTYPE} norm={NORM_TYPE} "
          f"channels={CHANNELS} sep_depth={SEP_DEPTH} volume={VOLUME} "
          f"tol={TOL:.0e}")
    torch.manual_seed(0)
    random.seed(0)

    if BACKBONE == "p1fl3":
        bb_cls, ref_cls = p1.BlockCNN_P1FL3, StandardUNetFL3
    elif BACKBONE == "memopt":
        bb_cls, ref_cls = BlockCNNMemopt, StandardUNet
    elif BACKBONE == "fl3":
        bb_cls, ref_cls = BlockCNNFL3, StandardUNetFL3Base
    else:
        bb_cls, ref_cls = BlockCNN, StandardUNet
    print(f"backbone={bb_cls.__name__}  reference={ref_cls.__name__}")
    bb = bb_cls(
        in_ch=IN_CH, channels=CHANNELS, blocks=BLOCKS, sep_depth=SEP_DEPTH,
        fusion_layers=FUSION_LAYERS, num_classes=NUM_CLASSES,
        use_aux_head=False, norm_type=NORM_TYPE,
    ).to(DEVICE).to(DTYPE)

    ref = ref_cls(
        in_ch=IN_CH, channels=CHANNELS, blocks=BLOCKS, sep_depth=SEP_DEPTH,
        fusion_layers=FUSION_LAYERS, num_classes=NUM_CLASSES,
        norm_type=NORM_TYPE,
    ).to(DEVICE).to(DTYPE)

    sync_weights(ref, bb)
    # the decoder must see the stag2 feature surrounded by R_stag3 zeros, as the
    # chunked path does (see ``StandardUNetFL3.f2halo``)
    if hasattr(ref, "f2halo"):
        ref.f2halo = int(getattr(bb, "_R_stag3", 0))
    inp, target = make_inputs()

    cases = []
    if max(VOLUME) <= 256:
        cases.append((1, 1, 1))  # whole-volume relay only fits at small volumes
    cases.append(tuple(CHUNK))
    if tuple(CHUNK) != (4, 4, 4):
        cases.append((4, 4, 4))

    failures = []
    if os.environ.get("ONLY_DICE", "0") != "1":
        for cs in cases:
            run_case(bb, ref, inp, target, cs, failures)

        # --- (b) EXTENDED CHECK: the production O(1) SEED path (Branch 2) -----
        # Re-run the same chunked relay but with a loss that supplies the whole-volume
        # seed analytically.  This exercises the path real training uses (with the
        # trainer's _grad_scale unscale emulated in run_case).  The whole-volume
        # reference is untouched (it always uses verify_loss_func), so a PASS here
        # proves the seed path is numerically identical to the reference backward.
        if os.environ.get("SEED", "1") != "0":
            seed_loss = SeedVerifyLossFunc()
            for cs in cases:
                run_case(bb, ref, inp, target, cs, failures,
                         cand_loss_func=seed_loss)

    # --- (c) DICE LOSS equivalence: same relay vs reference, Dice objective ----
    # The relay's whole-volume precompute differentiates the assembled prediction,
    # so the Dice term (a whole-volume sum) is reproduced exactly.  Gated by DICE=1
    # so the default (CE+BCE) run is unchanged.
    if os.environ.get("DICE", "0") == "1":
        dice_loss = verify_dice_loss_func
        for cs in cases:
            run_case(bb, ref, inp, target, cs, failures,
                     cand_loss_func=dice_loss, loss_fn=dice_loss)

    print("\n================ RESULT ================")
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures[:40]:
            print(f)
        raise SystemExit(1)
    print("  ALL CHECKS PASSED ✓ (loss + forward + per-param gradients "
          "match the whole-volume reference)")


if __name__ == "__main__":
    main()
