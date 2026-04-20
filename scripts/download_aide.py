import shutil
from pathlib import Path
import kagglehub

DATASET = "lburleigh/tla-lab-ai-detection-for-essays-aide-dataset"
DEST_DIR = Path("data/raw/aide")

def main():
    # Downloads to kagglehub cache dir; returns that path
    src_path = Path(kagglehub.dataset_download(DATASET))
    print("Kagglehub cache path:", src_path)

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy everything from the cache dir into your repo's data folder
    for p in src_path.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src_path)
            out = DEST_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

    print(f"Copied dataset files into: {DEST_DIR.resolve()}")
    print("Files:")
    for p in sorted(DEST_DIR.rglob("*")):
        if p.is_file():
            print(" -", p)

if __name__ == "__main__":
    main()
