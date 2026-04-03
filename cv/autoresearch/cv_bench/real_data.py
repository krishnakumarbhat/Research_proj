from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import requests
from PIL import Image
from scipy.io import loadmat

from cv_bench.common import DATA_CACHE


REAL_DATA_ROOT = DATA_CACHE / "real_datasets"
REQUEST_TIMEOUT = (30, 180)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
N_MNIST_ARCHIVE_URL = "https://data.mendeley.com/public-api/zip/468j46mzdv/download/1"
INDIAN_PINES_URLS = {
    "Indian_pines.mat": "https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
    "Indian_pines_gt.mat": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
}
CAMVID_BASE_RAW = "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid"
CAMVID_MANIFESTS = {
    "train.txt": 24,
    "val.txt": 8,
    "test.txt": 16,
}
DIBCO_REPO = "tanmayGIT/DIBCO_2019_All"
DIBCO_ORIGINAL_PATH = "DIBCO_2019_Work/DIBCO_2019_Dataset_1st_part/original"
DIBCO_GT_PATH = "DIBCO_2019_Work/DIBCO_2019_Dataset_1st_part/ground_truth_png"


def dataset_root(name: str) -> Path:
    root = REAL_DATA_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _github_directory(repo: str, path: str) -> list[dict[str, object]]:
    response = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"GitHub directory listing for {repo}/{path} did not return a list")
    return data


def _normalise_stem(name: str) -> str:
    stem = name.lower().strip()
    changed = True
    while changed:
        changed = False
        for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xcf"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)].strip()
                changed = True
    return stem


def _iter_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _sample_per_class(features: np.ndarray, labels: np.ndarray, *, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    chosen = []
    for label in sorted(np.unique(labels)):
        label_indices = np.flatnonzero(labels == label)
        take = min(len(label_indices), per_class)
        if take == len(label_indices):
            chosen.append(label_indices)
        else:
            chosen.append(rng.choice(label_indices, size=take, replace=False))
    if not chosen:
        return features, labels
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return features[indices], labels[indices]


def _load_rgb_image(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _load_grayscale_image(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask_raw(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
        return np.asarray(image)


def _load_binary_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = _load_mask_raw(path, target_hw)
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=-1)
    mask = mask.astype(np.float32)
    threshold = 127.5 if mask.max(initial=0.0) > 1.5 else 0.5
    foreground = mask < threshold if mask.mean() > threshold else mask > threshold
    return foreground.astype(np.int32)


def _first_mat_array(data: dict[str, object], ndim: int) -> np.ndarray:
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == ndim:
            return value
    raise ValueError(f"No {ndim}D array found in MAT file")


def _infer_n_mnist_split_and_label(path_name: str) -> tuple[str | None, str | None]:
    split = None
    label = None
    for part in PurePosixPath(path_name).parts:
        lower = part.lower()
        if lower in {"train", "test"}:
            split = lower
        if part.isdigit() and 0 <= int(part) <= 9:
            label = part
    return split, label


def _n_mnist_file_to_frame(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 5:
        return np.zeros((34, 34), dtype=np.float32)
    usable = raw[: raw.size - (raw.size % 5)].reshape(-1, 5)
    x = usable[:, 0].astype(np.int32)
    y = usable[:, 1].astype(np.int32)
    polarity = ((usable[:, 2] & 128) >> 7).astype(np.int32)
    valid = (x < 34) & (y < 34)
    if not np.any(valid):
        return np.zeros((34, 34), dtype=np.float32)
    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    frame = np.zeros((34, 34), dtype=np.float32)
    np.add.at(frame, (y, x), np.where(polarity > 0, 1.0, -1.0))
    frame = np.sign(frame) * np.log1p(np.abs(frame))
    frame -= frame.min(initial=0.0)
    peak = frame.max(initial=0.0)
    if peak > 0:
        frame /= peak
    return frame.astype(np.float32)


def _camvid_mask_to_coarse(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        indexed = mask.astype(np.int32)
        coarse = np.full(indexed.shape, 1, dtype=np.int32)
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 1,
            7: 1,
            8: 2,
            9: 2,
            10: 2,
        }
        for label, mapped in mapping.items():
            coarse[indexed == label] = mapped
        return coarse

    rgb = mask[..., :3].astype(np.uint8)
    coarse = np.full(rgb.shape[:2], 1, dtype=np.int32)
    color_mapping = {
        (128, 128, 128): 0,
        (128, 0, 0): 1,
        (192, 192, 128): 1,
        (192, 128, 128): 1,
        (64, 64, 128): 1,
        (128, 64, 128): 2,
        (0, 0, 192): 2,
        (64, 0, 128): 2,
        (64, 64, 0): 2,
        (128, 128, 0): 3,
    }
    for color, mapped in color_mapping.items():
        coarse[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = mapped
    return coarse


def _pair_image_and_mask_dirs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks_by_stem = {_normalise_stem(path.name): path for path in _iter_image_files(mask_dir)}
    pairs = []
    for image_path in _iter_image_files(image_dir):
        mask_path = masks_by_stem.get(_normalise_stem(image_path.name))
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def download_indian_pines() -> dict[str, str]:
    root = dataset_root("indian_pines")
    for file_name, url in INDIAN_PINES_URLS.items():
        target = root / file_name
        if not target.exists():
            _stream_download(url, target)
    return {
        "dataset": "Indian Pines",
        "status": "ready",
        "path": str(root),
        "notes": "Downloaded the hyperspectral cube and ground-truth labels.",
    }


def download_camvid_subset() -> dict[str, str]:
    root = dataset_root("camvid")
    raw_repo_root = "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master"
    downloaded = 0
    for manifest_name, limit in CAMVID_MANIFESTS.items():
        manifest_text = requests.get(f"{CAMVID_BASE_RAW}/{manifest_name}", timeout=REQUEST_TIMEOUT).text.splitlines()
        for line in manifest_text[:limit]:
            image_rel, mask_rel = line.split()
            for rel_path in [image_rel, mask_rel]:
                raw_rel = rel_path.replace("/SegNet", "").lstrip("/")
                local_rel = raw_rel.replace("CamVid/", "", 1)
                destination = root / local_rel
                if not destination.exists():
                    _stream_download(f"{raw_repo_root}/{raw_rel}", destination)
                    downloaded += 1
    return {
        "dataset": "CamVid subset",
        "status": "ready",
        "path": str(root),
        "notes": f"Downloaded a low-space GitHub subset with {sum(CAMVID_MANIFESTS.values())} image/mask pairs.",
    }


def download_dibco_subset() -> dict[str, str]:
    root = dataset_root("dibco")
    original_dir = root / "original"
    gt_dir = root / "ground_truth_png"
    original_items = _github_directory(DIBCO_REPO, DIBCO_ORIGINAL_PATH)
    gt_items = _github_directory(DIBCO_REPO, DIBCO_GT_PATH)
    gt_by_key = {_normalise_stem(str(item["name"])): item for item in gt_items}

    downloaded = 0
    for item in original_items:
        key = _normalise_stem(str(item["name"]))
        gt_item = gt_by_key.get(key)
        if gt_item is None:
            continue
        original_target = original_dir / str(item["name"])
        gt_target = gt_dir / str(gt_item["name"])
        if not original_target.exists():
            _stream_download(str(item["download_url"]), original_target)
        if not gt_target.exists():
            _stream_download(str(gt_item["download_url"]), gt_target)
        downloaded += 1
    return {
        "dataset": "DIBCO subset",
        "status": "ready",
        "path": str(root),
        "notes": f"Downloaded {downloaded} document/ground-truth pairs from a public GitHub mirror.",
    }


def download_n_mnist_subset(per_split_per_class: int = 40) -> dict[str, str]:
    root = dataset_root("n_mnist")
    existing_files = list(root.rglob("*.bin"))
    required_minimum = per_split_per_class * 20
    if len(existing_files) >= required_minimum:
        return {
            "dataset": "N-MNIST subset",
            "status": "ready",
            "path": str(root),
            "notes": f"Using existing extracted subset with {len(existing_files)} event files.",
        }

    archive_path = root / "n_mnist_download.zip"
    _stream_download(N_MNIST_ARCHIVE_URL, archive_path)
    counts: dict[tuple[str, str], int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = sorted((info for info in archive.infolist() if info.filename.lower().endswith(".bin")), key=lambda info: info.filename)
        for info in infos:
            split, label = _infer_n_mnist_split_and_label(info.filename)
            if split is None or label is None:
                continue
            key = (split, label)
            if counts.get(key, 0) >= per_split_per_class:
                continue
            destination = root / split / label / PurePosixPath(info.filename).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_handle, destination.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            counts[key] = counts.get(key, 0) + 1
    archive_path.unlink(missing_ok=True)
    extracted = sum(counts.values())
    return {
        "dataset": "N-MNIST subset",
        "status": "ready",
        "path": str(root),
        "notes": f"Downloaded the public Mendeley archive and extracted {extracted} binary event files before deleting the zip.",
    }


def download_supported_real_datasets(include_n_mnist: bool = True) -> list[dict[str, str]]:
    REAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    statuses = [download_indian_pines(), download_camvid_subset(), download_dibco_subset()]
    if include_n_mnist:
        statuses.append(download_n_mnist_subset())
    else:
        statuses.append(
            {
                "dataset": "N-MNIST subset",
                "status": "skipped",
                "path": str(dataset_root("n_mnist")),
                "notes": "Skipped the large public archive download by request.",
            }
        )
    return statuses


def load_n_mnist_images(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("n_mnist")
    files = sorted(root.rglob("*.bin"))
    if not files:
        return None
    per_class = 20 if quick else 60
    class_counts: dict[str, int] = {}
    selected: list[tuple[Path, int]] = []
    for path in files:
        label = next((part for part in path.parts if part.isdigit() and 0 <= int(part) <= 9), None)
        if label is None:
            continue
        if class_counts.get(label, 0) >= per_class:
            continue
        class_counts[label] = class_counts.get(label, 0) + 1
        selected.append((path, int(label)))
    if not selected:
        return None
    images = np.asarray([_n_mnist_file_to_frame(path) for path, _ in selected], dtype=np.float32)
    labels = np.asarray([label for _, label in selected], dtype=np.int32)
    return images, labels, "n_mnist_real_subset"


def load_indian_pines_spectra(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("indian_pines")
    cube_path = root / "Indian_pines.mat"
    gt_path = root / "Indian_pines_gt.mat"
    if not cube_path.exists() or not gt_path.exists():
        return None
    cube = _first_mat_array(loadmat(cube_path), 3).astype(np.float32)
    gt = _first_mat_array(loadmat(gt_path), 2).astype(np.int32)
    labeled = gt > 0
    if not np.any(labeled):
        return None
    spectra = cube[labeled].reshape(-1, cube.shape[-1]).astype(np.float32)
    labels = gt[labeled].reshape(-1).astype(np.int32) - 1
    spectra_min = spectra.min(axis=0, keepdims=True)
    spectra_range = np.maximum(spectra.max(axis=0, keepdims=True) - spectra_min, 1e-6)
    spectra = (spectra - spectra_min) / spectra_range
    spectra, labels = _sample_per_class(spectra, labels, per_class=35 if quick else 90)
    return spectra, labels, "indian_pines_real"


def load_camvid_images(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("camvid")
    pairs = []
    for split in ["train", "val", "test"]:
        pairs.extend(_pair_image_and_mask_dirs(root / split, root / f"{split}annot"))
    if not pairs:
        return None
    limit = min(len(pairs), 18 if quick else 48)
    images = []
    masks = []
    for image_path, mask_path in pairs[:limit]:
        images.append(_load_rgb_image(image_path, (48, 64)))
        masks.append(_camvid_mask_to_coarse(_load_mask_raw(mask_path, (48, 64))))
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.int32), "camvid_github_subset"


def load_dibco_documents(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("dibco")
    pairs = _pair_image_and_mask_dirs(root / "original", root / "ground_truth_png")
    if not pairs:
        return None
    limit = min(len(pairs), 5 if quick else 5)
    images = []
    masks = []
    for image_path, mask_path in pairs[:limit]:
        images.append(_load_grayscale_image(image_path, (48, 64)))
        masks.append(_load_binary_mask(mask_path, (48, 64)))
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.int32), "dibco_github_subset"


def load_busi_images(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("busi")
    if not root.exists():
        return None
    image_paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "mask" not in path.stem.lower()]
    if not image_paths:
        return None
    limit = min(len(image_paths), 20 if quick else 60)
    images = []
    masks = []
    for image_path in sorted(image_paths)[:limit]:
        mask_candidates = [
            image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
            image_path.with_name(f"{image_path.stem}_mask.png"),
        ]
        mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
        images.append(_load_grayscale_image(image_path, (48, 48)))
        if mask_path is None:
            masks.append(np.zeros((48, 48), dtype=np.int32))
        else:
            masks.append(_load_binary_mask(mask_path, (48, 48)))
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.int32), "busi_local"


def load_kvasir_depth_pairs(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("kvasir_depth")
    image_dir_candidates = [root / "images", root / "rgb"]
    depth_dir_candidates = [root / "depths", root / "depth", root / "gt"]
    image_dir = next((candidate for candidate in image_dir_candidates if candidate.exists()), None)
    depth_dir = next((candidate for candidate in depth_dir_candidates if candidate.exists()), None)
    if image_dir is None or depth_dir is None:
        return None
    depth_by_stem = {_normalise_stem(path.name): path for path in _iter_image_files(depth_dir)}
    images = []
    depths = []
    limit = 12 if quick else 32
    for image_path in _iter_image_files(image_dir):
        depth_path = depth_by_stem.get(_normalise_stem(image_path.name))
        if depth_path is None:
            continue
        images.append(_load_rgb_image(image_path, (40, 48)))
        depth = _load_grayscale_image(depth_path, (40, 48))
        depths.append(depth.astype(np.float32))
        if len(images) >= limit:
            break
    if not images:
        return None
    return np.asarray(images, dtype=np.float32), np.asarray(depths, dtype=np.float32), "kvasir_depth_local"


def load_mvtec_single_category(quick: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    root = dataset_root("mvtec")
    category_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    category_dir = next((path for path in category_dirs if (path / "train").exists() and (path / "test").exists()), None)
    if category_dir is None:
        return None
    train_good = _iter_image_files(category_dir / "train" / "good")
    if not train_good:
        return None
    train_limit = min(len(train_good), 18 if quick else 48)
    train_images = np.asarray([_load_rgb_image(path, (40, 40)) for path in train_good[:train_limit]], dtype=np.float32)

    test_images = []
    labels = []
    masks = []
    for defect_dir in sorted(path for path in (category_dir / "test").iterdir() if path.is_dir()):
        for image_path in _iter_image_files(defect_dir):
            test_images.append(_load_rgb_image(image_path, (40, 40)))
            is_good = defect_dir.name == "good"
            labels.append(0 if is_good else 1)
            if is_good:
                masks.append(np.zeros((40, 40), dtype=np.int32))
            else:
                mask_name = f"{image_path.stem}_mask.png"
                mask_path = category_dir / "ground_truth" / defect_dir.name / mask_name
                masks.append(_load_binary_mask(mask_path, (40, 40)) if mask_path.exists() else np.zeros((40, 40), dtype=np.int32))
            if len(test_images) >= (20 if quick else 60):
                break
        if len(test_images) >= (20 if quick else 60):
            break
    if not test_images:
        return None
    return (
        train_images,
        np.asarray(test_images, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(masks, dtype=np.int32),
        f"mvtec_local_{category_dir.name}",
    )


def load_climate_patches(quick: bool) -> tuple[np.ndarray, np.ndarray, str] | None:
    root = dataset_root("planet")
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return None
    mask_by_stem = {_normalise_stem(path.name): path for path in _iter_image_files(mask_dir)}
    images = []
    masks = []
    limit = 20 if quick else 48
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        key = _normalise_stem(image_path.name)
        mask_path = mask_by_stem.get(key)
        if mask_path is None:
            continue
        if image_path.suffix.lower() == ".npy":
            image = np.load(image_path).astype(np.float32)
            if image.ndim != 3 or image.shape[-1] not in {4, 5}:
                continue
            image = image[..., :4]
            image = np.clip(image, 0.0, 1.0)
            pil = Image.fromarray((image[..., :3] * 255).astype(np.uint8))
            rgb = np.asarray(pil.resize((48, 48), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            nir = np.asarray(Image.fromarray((image[..., 3] * 255).astype(np.uint8)).resize((48, 48), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            image = np.concatenate([rgb, nir[..., None]], axis=-1)
        else:
            continue
        mask = _load_binary_mask(mask_path, (48, 48))
        images.append(image)
        masks.append(mask)
        if len(images) >= limit:
            break
    if not images:
        return None
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.int32), "planet_local_subset"