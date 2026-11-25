"""Install the NVIDIA CUDA libraries used by the DHD JTorch environment."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory


NVIDIA = "https://developer.download.nvidia.com/compute"
CUDA_COMPONENTS = (
    "cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas", "libcufft",
    "libcurand", "libcusolver", "libcusparse", "libnvjitlink", "libnpp",
)
CURL = ["curl", "-fLsS", "--retry", "3", "--connect-timeout", "20"]


def manifest(prefix, version):
    return json.loads(subprocess.check_output([
        *CURL, f"{NVIDIA}/{prefix}/redist/redistrib_{version}.json",
    ]))


def checksum(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def download(cache, component):
    prefix, metadata = component
    relative_path = metadata["relative_path"]
    target = cache / Path(relative_path).name
    if target.is_file() and checksum(target) == metadata["sha256"]:
        print(f"Using verified {target.name}", flush=True)
        return target

    print(f"Downloading {target.name}", flush=True)
    with NamedTemporaryFile(dir=cache, delete=False) as temporary:
        part = Path(temporary.name)
    try:
        subprocess.run([*CURL, "-o", str(part), f"{NVIDIA}/{prefix}/redist/{relative_path}"], check=True)
        if checksum(part) != metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {target.name}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target


def install(target, cache):
    cache.mkdir(parents=True, exist_ok=True)
    cuda = manifest("cuda", "12.8.1")
    cudnn = manifest("cudnn", "8.9.7")
    components = [("cuda", cuda[key]["linux-x86_64"]) for key in CUDA_COMPONENTS]
    components.append(("cudnn", cudnn["cudnn"]["linux-x86_64"]["cuda12"]))
    with TemporaryDirectory(dir=target.parent, prefix="cuda-") as temporary:
        stage = Path(temporary) / "toolkit"
        stage.mkdir()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for archive in executor.map(lambda component: download(cache, component), components):
                print(f"Extracting {archive.name}", flush=True)
                subprocess.run([
                    "tar", "-xJf", str(archive), "--strip-components=1", "-C", str(stage),
                ], check=True)
        (stage / "lib64").symlink_to("lib", target_is_directory=True)
        stage.rename(target)


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
