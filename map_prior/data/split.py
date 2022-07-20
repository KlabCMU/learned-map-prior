from pathlib import Path
from ..config import Mode, Split


Building = str
Dataset = dict[Building, dict[Mode, list[Path]]]


def split(dir: Path) -> Dataset:
    files: Dataset = dict()
    for file in sorted(dir.glob("*building*/**/*.npy")):
        building: Building = file.parents[1].name
        split = Split(file.parents[0].name)
        if building not in files:
            files[building] = dict()
        if split not in files[building]:
            files[building][split] = list()
        files[building][split].append(file)

    return files
