# Jade Oakes
# Download DAIGT V2 dataset (clean project copy)

import shutil
from pathlib import Path
import kagglehub


DATASET = "thedrcat/daigt-v2-train-dataset"
DEST_DIR = Path("data/raw/daigt_v2")


def main():
    print("Downloading DAIGT V2 from KaggleHub...")
    src_path = Path(kagglehub.dataset_download(DATASET))
    print("KaggleHub cache path:", src_path)

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files into project directory
    for p in src_path.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src_path)
            out = DEST_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

    print(f"Copied dataset into: {DEST_DIR.resolve()}")

    print("Files:")
    for p in DEST_DIR.rglob("*"):
        if p.is_file():
            print(" -", p)


if __name__ == "__main__":
    main()
