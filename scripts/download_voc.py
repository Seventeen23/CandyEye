import argparse
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

files = {
   "trainval": {
        "url": (
            "https://www.robots.ox.ac.uk/~vgg/projects/pascal/"
            "VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
        ),
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "expected": {
            "trainval.txt": 5011
        },
    },
    "test": {
        "url": (
            "https://www.robots.ox.ac.uk/~vgg/projects/pascal/"
            "VOC/voc2007/VOCtest_06-Nov-2007.tar"
        ),
        "filename": "VOCtest_06-Nov-2007.tar",
        "expected": {
            "test.txt": 4952,
        },
    },
}

class Tqdm_Download(tqdm):

    # Progress bar for downloading files

    def __init__(self, total, filename):
        self.progress = tqdm(
            total = total,
            unit = "B",
            unit_scale = True, 
            unit_divisor = 1024,
            desc = filename,

        )

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0 and self.progress.total != total_size:
            self.progress.total = total_size

        downloaded = block_num * block_size

        self.progress.update(max(0, downloaded - self.progress.n))

    def close(self):
        self.progress.close()


def download_file(url: str, destination: Path) -> None:

    """Download a file with tqdm progress."""

    if destination.exists():
        print(f"⏭  Already exists: {destination}")
        return

    print(f"⬇  Downloading: {destination.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    progress = None

    try:
        # Get the file size first so tqdm has an accurate total.
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}, # Change the User-Agent if there are 4XX errs
        )

        with urllib.request.urlopen(request) as response:
            total_size = response.headers.get("Content-Length")

            if total_size is not None:
                total_size = int(total_size)

            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            )

            with open(destination, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)
                    progress.update(len(chunk))

    except Exception:
        # Don't leave a partially downloaded archive behind.
        if destination.exists():
            destination.unlink()

        raise

    finally:
        if progress is not None:
            progress.close()

    print(f"Downloaded: {destination.name}")
def extract_tar(tar_path: Path, output_dir: Path) -> None:
    """Safely extract a tar archive."""

    print(f" Extracting: {tar_path.name}")

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(output_dir, filter="data")

    print(f" Extracted: {tar_path.name}")


def verify_file(path: Path, expected_lines: int) -> bool:
    """Verify that a text file contains the expected number of lines."""

    if not path.exists():
        print(f" Missing: {path}")
        return False

    with path.open("r", encoding="utf-8") as f:
        line_count = sum(1 for line in f)

    if line_count == expected_lines:
        print(f" {path}: {line_count} lines")
        return True

    print(
        f" {path}: expected {expected_lines} lines, "
        f"found {line_count}"
    )
    return False


def verify_voc(output_dir: Path, skip_test: bool) -> bool:
    """Verify the extracted VOC 2007 dataset."""

    main_dir = (
        output_dir
        / "VOCdevkit"
        / "VOC2007"
        / "ImageSets"
        / "Main"
    )

    trainval_ok = verify_file(
        main_dir / "trainval.txt",
        5011,
    )

    if skip_test:
        return trainval_ok

    test_ok = verify_file(
        main_dir / "test.txt",
        4952,
    )

    return trainval_ok and test_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract PASCAL VOC 2007."
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data)",
    )

    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep downloaded tar files after successful verification",
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip downloading and verifying the VOC test set",
    )

    args = parser.parse_args()

    out_dir = args.out
    downloads_dir = out_dir / "downloads"

    downloads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------

    trainval_tar = downloads_dir / files["trainval"]["filename"]

    download_file(
        files["trainval"]["url"],
        trainval_tar,
    )

    test_tar = downloads_dir / files["test"]["filename"]

    if not args.skip_test:
        download_file(
            files["test"]["url"],
            test_tar,
        )

    # ------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------

    extract_tar(trainval_tar, out_dir)

    if not args.skip_test:
        extract_tar(test_tar, out_dir)

    # ------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------

    print("\n🔍 Verifying dataset...")

    verified = verify_voc(
        out_dir,
        skip_test=args.skip_test,
    )

    if not verified:
        print("\n Verification failed. Keeping tar files.")
        return 1

    print("\n VOC 2007 verification successful.")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    if not args.keep_tar:
        print("\n Removing tar files...")

        trainval_tar.unlink(missing_ok=True)

        if not args.skip_test:
            test_tar.unlink(missing_ok=True)

        print(" Tar files removed.")
    else:
        print("\n Keeping tar files (--keep-tar).")

    print(f"\n VOC 2007 dataset ready at: {out_dir / 'VOCdevkit' / 'VOC2007'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())