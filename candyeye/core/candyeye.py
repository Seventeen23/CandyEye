"""YAML-driven model builder + forward graph for the YOLO11n detector.

The YAML in ``configs/yolo11.yaml`` is the single source of truth for the
architecture. The builder walks it exactly like the official implementation:

- each entry is ``[from, repeats, module, args]``
- ``from`` < 0  -> offset into the built list (-1 = previous layer)
- ``from`` >= 0 -> absolute layer index; a list = several layers (Concat / Detect)
- output channels are tracked alongside the built layers
- width : ``c2 = make_divisible(min(c2, max_channels) * width_mult, 8)``
- depth : ``repeats = max(round(repeats * depth_mult), 1)``

The result is an ``nn.Sequential`` indexed exactly like the official yolo11n
(0..23), so state_dict keys land at ``model.0.conv.weight``,
``model.23.cv3.1.0.1.conv.weight``, ... and Phase 2 can load the official
``yolo11n.pt`` 1:1 after re-initialising the head's class branch for nc=20.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import torch
from torch import nn

from candyeye.core.functions.layer_utils import make_divisible
from candyeye.core.modules.blocks import Attention, Bottleneck, C2PSA, C3k, C3k2, PSABlock, SPPF
from candyeye.core.modules.conv import Conv
from candyeye.core.modules.detect import Detect
from candyeye.core.modules.exchange import ScaleExchange
from candyeye.paths import resolve_config


class Concat(nn.Module):
    """Concatenate several tensors along a dimension (neck merges)."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


_MODULE_MAP = {
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "C3k": C3k,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "Attention": Attention,
    "PSABlock": PSABlock,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
    "ScaleExchange": ScaleExchange,
    "nn.Upsample": nn.Upsample,
    "nn.Conv2d": nn.Conv2d,
}

_CHANNEL_MODULES = {"Conv", "C3k2", "C2PSA", "SPPF", "C3k", "Bottleneck", "Attention", "PSABlock"}

# depth_mult, width_mult, max_channels
_SCALES = {"n": (0.50, 0.25, 1024)}


def _source_channels(src: list[int], cur: int | list[int],
                     ch: dict[int, int | list[int]]) -> list[int]:
    """Flatten the input-channel spec of a multi-input layer.

    A source entry may itself be a channel *list* (a previous multi-output
    layer such as ``ScaleExchange``), in which case its channels are spliced in.
    """
    channels: list[int] = []
    for index in src:
        entry = cur if index == -1 else ch[index]
        if isinstance(entry, (list, tuple)):
            channels.extend(entry)
        else:
            channels.append(entry)
    return channels



def parse_model(data: dict) -> nn.Sequential:
    """Expand the yaml layer table into an indexed nn.Sequential."""
    nc = data["nc"]
    scale = data.get("scale", "n")
    if scale not in _SCALES:
        raise ValueError(f"scale {scale!r} not supported (have {sorted(_SCALES)})")
    depth, width, max_channels = _SCALES[scale]

    x = data["backbone"] + data["head"]
    layers, ch, cur = [], {}, 3  # ch[i] = out channels of layer i; cur = last output

    for i, (f, n, m, args) in enumerate(x):
        args = [nc if a == "nc" else a for a in args]
        for j, a in enumerate(args):
            if isinstance(a, str) and a != "nc":
                try:
                    args[j] = ast.literal_eval(a)  # "None" -> None, keep "nearest"
                except (ValueError, SyntaxError):
                    pass
        n = max(round(n * depth), 1)

        if m in _CHANNEL_MODULES:
            c1, c2 = cur, args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m == "C3k2":
                args.insert(2, n)  # repeats slot inside the block
                n = 1
        elif m == "nn.Upsample":
            c2 = cur
        elif m == "Concat":
            c2 = sum(cur if x == -1 else ch[x] for x in f)
        elif m == "Detect":
            src = f if isinstance(f, (list, tuple)) else [f]
            args.append(_source_channels(src, cur, ch))
            c2 = nc  # not referenced downstream; placeholder
        elif m == "ScaleExchange":
            src = f if isinstance(f, (list, tuple)) else [f]
            in_ch = _source_channels(src, cur, ch)
            gate = str(args[0]) if args else "none"
            iters = int(args[1]) if len(args) > 1 else 1
            args = [*in_ch, gate, iters]
            c2 = list(in_ch)  # this layer emits a channel *list*
        else:
            raise ValueError(f"unhandled module {m!r}")

        module_cls = _MODULE_MAP[m]
        if m == "Detect":
            module = module_cls(nc=args[0], ch=args[1])
        elif n > 1:
            module = nn.Sequential(*(copy.deepcopy(module_cls(*args)) for _ in range(n)))
        else:
            module = module_cls(*args)
        module.type = m
        module.i = i  # absolute index
        module.f = f  # from-list (int or list of ints)
        module.rep = n  # repeats used (rep avoids clobbering SPPF/C2PSA self.n)

        layers.append(module)
        ch[i] = c2
        cur = c2
    return nn.Sequential(*layers)


class CandyEye(nn.Module):
    """CandyEye detector: YAML-built graph and anchor-free Detect head."""

    def __init__(self, cfg=None, nc: int | None = None, img_size: int = 128):
        super().__init__()
        resolved = cfg if isinstance(cfg, dict) else resolve_config(cfg)
        data = copy.deepcopy(resolved) if isinstance(resolved, dict) else self._load_yaml(resolved)
        if nc is not None:
            data["nc"] = nc
        self.model = parse_model(data)
        self.nc = self.model[-1].nc
        self.yaml = resolved
        self.img_size = img_size
        if img_size % 32:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.stride = self._detect_stride(img_size)
        self.model[-1].stride = self.stride
        self.model[-1].bias_init()

    def set_classes(self, nc: int):
        """Resize the detection class head, preserving all compatible weights."""
        nc = int(nc)
        if nc <= 0:
            raise ValueError(f"nc must be positive, got {nc}")
        old = self.model[-1]
        if nc == old.nc:
            self.nc = nc
            return self

        channels = tuple(branch[0].conv.in_channels for branch in old.cv2)
        new = Detect(nc=nc, reg_max=old.reg_max, ch=channels).to(
            device=old.stride.device, dtype=next(old.parameters()).dtype)
        new.stride = old.stride.clone()
        new.bias_init()
        for attribute in ("type", "i", "f", "rep"):
            if hasattr(old, attribute):
                setattr(new, attribute, getattr(old, attribute))
        old_state, new_state = old.state_dict(), new.state_dict()
        with torch.no_grad():
            for key, value in new_state.items():
                if key in old_state and old_state[key].shape == value.shape:
                    value.copy_(old_state[key])
        self.model[-1] = new
        self.nc = nc
        return self

    @staticmethod
    def _load_yaml(cfg: str) -> dict:
        import yaml

        with open(cfg) as fh:
            data = yaml.safe_load(fh)
        CandyEye._validate_yaml(data)
        return data

    @staticmethod
    def _validate_yaml(data: dict) -> None:
        path = Path(data.get("path", "?"))
        for section in ("backbone", "head"):
            if section not in data:
                raise ValueError(f"yaml {path} missing section {section!r}")
            for entry in data[section]:
                if not (isinstance(entry, list) and len(entry) == 4):
                    raise ValueError(f"bad {section} entry {entry!r} (want [from, repeats, module, args])")
        if "nc" not in data:
            raise ValueError(f"yaml {path} missing nc")

    def _detect_stride(self, img_size: int) -> torch.Tensor:
        """Feed a blank image and infer P3/P4/P5 strides from output sizes."""
        was_training = self.training
        self.eval()  # BatchNorm train mode rejects 1x1 feature maps (img_size=32)
        try:
            with torch.no_grad():
                # decode=False: raw per-level maps regardless of train/eval.
                feats = self(torch.zeros(1, 3, img_size, img_size), decode=False)
        finally:
            if was_training:
                self.train()
        if not isinstance(feats, (list, tuple)):
            return torch.ones(1)
        return torch.tensor([img_size / f.shape[-2] for f in feats])

    def forward(self, x: torch.Tensor, *, decode: bool | None = None) -> torch.Tensor | list[torch.Tensor]:
        """Run the graph; ``decode=False`` returns raw maps even in eval mode."""
        y = []  # layer-output history, index == absolute layer index
        for m in self.model:
            f = m.f
            if isinstance(f, int):
                xi = x if f == -1 else y[f]
            else:
                xi = [x if j == -1 else y[j] for j in f]
            x = m(xi, decode=decode) if isinstance(m, Detect) else m(xi)
            y.append(x)
        return x

    def train(self, mode: bool = True, *, data=None, epochs: int = 100,
              imgsz: int | None = None, batch: int = 16, patience: int = 50,
              workers: int = 0, device: str = "cpu",
              project: str | Path = "runs/train", name: str = "exp",
              resume: bool | str | Path = False, optimizer: str = "AdamW",
              lr0: float = 2e-4, weight_decay: float = 5e-4,
              warmup_epochs: float = 3, mosaic: float = .5,
              hsv: bool = True, fliplr: float = .5,
              pretrained: bool | str | Path = False, seed: int = 23,
              exist_ok: bool = False, max_batches: int | None = None,
              threads: int = 4):
        """Set PyTorch mode or launch CandyEye training when `data` is set.

        Example: ``model.train(data="configs/default.yaml", epochs=100,
        imgsz=128, batch=16, patience=20)``. When called without `data`, this
        retains the standard ``nn.Module.train(mode)`` behavior.
        """
        if data is None:
            return super().train(mode)
        from candyeye.training.trainer import train_model

        return train_model(
            self, data=data, epochs=epochs, imgsz=imgsz or self.img_size,
            batch=batch, patience=patience, workers=workers, device=device,
            project=project, name=name, resume=resume, optimizer=optimizer,
            lr0=lr0, weight_decay=weight_decay, warmup_epochs=warmup_epochs,
            mosaic=mosaic, hsv=hsv, fliplr=fliplr, pretrained=pretrained,
            seed=seed, exist_ok=exist_ok, max_batches=max_batches,
            threads=threads,
        )

    def fuse(self):
        """Fold every BatchNorm into its conv (in place).

        Puts the model in eval mode first (BN running stats are the whole
        point of fusing).  After this, state_dict keys match the ONNX
        versions of the model (``model.0.conv.weight`` +
        ``model.0.conv.bias``, no ``.bn.*``), and ``load_fused_from_onnx``
        can copy the official fp32 weights 1:1.  Numerical parity with the
        official export is then exact, not "fp16 checkpoint drift" close.
        """
        from candyeye.core.modules.conv import Conv

        self.eval()
        for m in self.modules():
            if isinstance(m, Conv):
                m.fuse()
        return self
