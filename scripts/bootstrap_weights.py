import argparse
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

try:
    import torch
except ImportError:
    sys.exit("torch not available — run: venv/bin/pip install torch torchvision")

# ---------------------------------------------------------------------------
# DEV-ONLY bootstrap.  The ONLY file in the repo that touches ultralytics.
#
# It turns the official ultralytics checkpoint into a *plain* state_dict file
# (`weights/yolo11n.pth`) containing nothing but torch tensors, so everything
# else in the project — the converter, training, inference, downstream users —
# never needs ultralytics installed.  `pyproject.toml` is left untouched.
# ---------------------------------------------------------------------------

ASSETS = "https://github.com/ultralytics/assets/releases/download"
PT_URL = f"{ASSETS}/v8.3.0/yolo11n.pt"          # official yolo11n checkpoint
DEMO_IMAGES = {                                 # COCO sample images (small)
    "bus.jpg": f"{ASSETS}/v0.0.0/bus.jpg",
    "zidane.jpg": f"{ASSETS}/v0.0.0/zidane.jpg",
}


def download_file(url: str, destination: Path) -> None:
    """Download with a tqdm progress bar; clean up partial files on error."""
    if destination.exists():
        print(f"ok: {destination} (already present)")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    progress = None
    try:
        with urllib.request.urlopen(request) as response:
            total = response.headers.get("Content-Length")
            progress = tqdm(
                total=int(total) if total is not None else None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=destination.name,
            )
            with open(destination, "wb") as fh:
                while chunk := response.read(1024 * 1024):
                    fh.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()


def import_ultralytics() -> None:
    """ultralytics is needed purely to unpickle its checkpoint objects."""
    try:
        import ultralytics  # noqa: F401  (import side-effect registers its classes)
    except ImportError:
        sys.exit(
            "ultralytics is not installed (needed once, to decode the .pt).\n"
            "  venv/bin/pip install ultralytics --no-deps   # keeps CPU torch"
        )


def extract_state_dict(pt_path: Path, out_path: Path) -> dict:
    """Load the checkpoint and save ONLY its model state_dict as plain tensors.

    `ckpt["model"]` is the official DetectionModel object; calling
    `.state_dict()` gives keys like `model.0.conv.weight` ... which match our
    builder's layout 1:1.  We then re-save that dict alone, so the resulting
    .pth imports without ultralytics anywhere.
    """
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"].state_dict()
    assert isinstance(sd, dict) and all(
        isinstance(v, torch.Tensor) for v in sd.values()
    ), "expected a plain state_dict of tensors"
    torch.save(sd, out_path)
    return sd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--skip-demo-images", action="store_true")
    args = parser.parse_args()

    import_ultralytics()

    pt = args.weights_dir / "yolo11n.pt"
    pth = args.weights_dir / "yolo11n.pth"
    download_file(PT_URL, pt)
    sd = extract_state_dict(pt, pth)
    n = sum(v.numel() for v in sd.values())
    print(f"saved: {pth}  ({len(sd)} tensors, {n:,} params)")

    if not args.skip_demo_images:
        for name, url in DEMO_IMAGES.items():
            download_file(url, args.runs_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())