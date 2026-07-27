"""Download the fixed Gierer–Meinhardt train/validation/evaluation data."""

from pathlib import Path
import shutil
import tempfile

import gdown


URL = (
    "https://drive.google.com/drive/folders/"
    "1kFGnLe4YJ6ddM5YCSfyXyZ86eCBLJb96?usp=sharing"
)
OUTPUT_DIR = Path("gm_data")
FILES = {
    "train_data.npz": "gm_train_data.npz",
    "val_data.npz": "gm_validation_data.npz",
    "test_data.npz": "gm_eval_data.npz",
}


def download_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        OUTPUT_DIR / destination
        for destination in FILES.values()
        if (OUTPUT_DIR / destination).exists()
    ]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing GM data: {names}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=".gm_download_", dir=".")
    )
    try:
        gdown.download_folder(
            URL,
            output=str(temporary),
            quiet=False,
            use_cookies=False,
        )
        for source_name, destination_name in FILES.items():
            matches = list(temporary.rglob(source_name))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one {source_name}, found {len(matches)}."
                )
            matches[0].replace(OUTPUT_DIR / destination_name)
    finally:
        shutil.rmtree(temporary)

    print(f"GM dataset is ready in {OUTPUT_DIR}/")


if __name__ == "__main__":
    download_dataset()
