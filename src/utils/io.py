import os
import os.path as osp
from pathlib import Path


def _get_next_version(base_dir) -> int:
    versions_root = Path(base_dir)
    versions_root.mkdir(parents=True, exist_ok=True)

    if not versions_root.is_dir():
        print("Missing logger folder: %s", versions_root)
        return 0

    existing_versions = []
    for d in versions_root.iterdir():
        if d.is_dir() and d.name.startswith("version_"):
            dir_ver = d.name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))

    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def prepare_version_dir(base_dir, mkdir=False):
    version = _get_next_version(base_dir)
    version_dir = osp.join(base_dir, f"version_{version}")
    if mkdir:
        os.makedirs(version_dir, exist_ok=True)
    return version_dir
