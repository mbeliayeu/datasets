#!/usr/bin/env python3
"""Mirror every file in an HF dataset repository into bounded release assets."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


class ArchiveParts:
    """Split an output stream without keeping a second, unsplit archive on disk."""

    def __init__(self, directory, prefix, max_bytes, max_parts=998):
        if max_bytes < 1 or max_parts < 1:
            raise ValueError("Archive limits must be positive")
        self.directory = directory
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.max_parts = max_parts  # Reserve two assets for manifest and checksums.
        self.parts = []
        self.current = None

    def write(self, data):
        remaining = memoryview(data)
        while remaining:
            if self.current is None:
                if len(self.parts) >= self.max_parts:
                    raise ValueError("Too many archive parts for one GitHub release")
                name = f"{self.prefix}.tar.gz.part{len(self.parts) + 1:04d}"
                self.current = (self.directory / name).open("wb")
                self.digest = hashlib.sha256()
                self.size = 0
            chunk = remaining[: self.max_bytes - self.size]
            self.current.write(chunk)
            self.digest.update(chunk)
            self.size += len(chunk)
            remaining = remaining[len(chunk):]
            if self.size == self.max_bytes:
                self.close()
        return len(data)

    def flush(self):
        if self.current is not None:
            self.current.flush()

    def close(self):
        if self.current is not None:
            self.current.close()
            self.parts.append({
                "name": Path(self.current.name).name,
                "bytes": self.size,
                "sha256": self.digest.hexdigest(),
            })
            self.current = None


def make_archive(snapshot, file_names, output_dir, prefix, max_bytes):
    writer = ArchiveParts(output_dir, prefix, max_bytes)
    files = []
    try:
        with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", dereference=True) as archive:
                for name in sorted(file_names):
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Unsafe dataset path: {name!r}")
                    source = snapshot / name
                    if not source.resolve().is_relative_to(snapshot.resolve()) or not source.is_file():
                        raise ValueError(f"Missing or unsafe dataset file: {name!r}")
                    # Only Hub-tracked files are archived, excluding local download metadata.
                    archive.add(source, arcname=f"{prefix}/{name}", recursive=False)
                    files.append({"path": name, "bytes": source.stat().st_size})
    finally:
        writer.close()
    if len(writer.parts) == 1:
        part = writer.parts[0]
        name = f"{prefix}.tar.gz"
        (output_dir / part["name"]).rename(output_dir / name)
        part["name"] = name
    return files, writer.parts


def prepare_dataset(dataset_id, revision, work_dir, asset_size_mib=1900, max_workers=4,
                    release_tag=""):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(dataset_id)
    if not 1 <= asset_size_mib <= 1900:
        raise ValueError("asset_size_mib must be between 1 and 1900")
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    if not revision.strip():
        raise ValueError("revision must not be blank")
    if release_tag:
        if release_tag.startswith("-"):
            raise ValueError("release_tag must not start with '-'")
        subprocess.run(["git", "check-ref-format", f"refs/tags/{release_tag}"], check=True)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "assets"
    output_dir.mkdir(exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Asset directory must be empty: {output_dir}")

    token = os.environ.get("HF_TOKEN") or False
    info = HfApi().dataset_info(dataset_id, revision=revision, files_metadata=True, token=token)
    if not info.sha or not info.siblings:
        raise ValueError("Dataset revision has no commit SHA or files")
    file_names = [sibling.rfilename for sibling in info.siblings]
    # Check space for source + archive, allowing for gzip/tar overhead and metadata.
    sizes = [sibling.size for sibling in info.siblings]
    if all(size is not None for size in sizes):
        required = 2 * sum(sizes) + 512 * 1024**2
        available = shutil.disk_usage(work_dir).free
        if required > available:
            raise ValueError(
                f"Insufficient disk: need about {required / 1024**3:.1f} GiB; "
                f"{available / 1024**3:.1f} GiB available. Use a larger runner."
            )

    prefix = f"{dataset_id.replace('/', '--')}-{info.sha[:12]}"
    run = os.environ.get("GITHUB_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    tag = release_tag or f"hf-{prefix}-{run}-{attempt}"
    print(f"Downloading all {len(file_names)} files from {dataset_id} @ {info.sha}", flush=True)
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=work_dir) as temporary:
        snapshot = Path(snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=info.sha,
            local_dir=temporary,
            max_workers=max_workers,
            token=token,
        ))
        files, parts = make_archive(snapshot, file_names, output_dir, prefix, asset_size_mib * 1024**2)

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "requested_revision": revision,
        "commit_sha": info.sha,
        "source_url": f"https://huggingface.co/datasets/{dataset_id}/tree/{info.sha}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "Hugging Face repository snapshot (tar.gz; concatenate parts in name order)",
        "archive_root": prefix,
        "files": files,
        "archive_parts": parts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = [f"{part['sha256']}  {part['name']}\n" for part in parts]
    checksums.append(f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  manifest.json\n")
    (output_dir / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    extraction = (
        f"tar -xzf {prefix}.tar.gz" if len(parts) == 1 else
        f"cat {prefix}.tar.gz.part* | tar -xz"
    )
    notes = (
        f"Complete repository snapshot of [{dataset_id}]({manifest['source_url']}).\n\n"
        f"Hugging Face commit: `{info.sha}`. Includes all {len(files)} repository files, "
        "covering every stored subset and split, plus dataset documentation and license files.\n\n"
        "Download all archive assets, `manifest.json`, and `SHA256SUMS` into one directory.\n\n"
        f"```sh\nsha256sum -c SHA256SUMS\n{extraction}\n```\n\n"
        f"Files extract into `{prefix}/`. Original formats and directory structure are preserved. "
        "External URLs are preserved as references; data hosted outside the HF repository is not fetched. "
        "The dataset's original license and attribution requirements apply.\n"
    )
    (work_dir / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")
    result = {"tag": tag, "title": f"{dataset_id} @ {info.sha[:12]}"}
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for name, value in result.items():
                output.write(f"{name}={value}\n")
    print(json.dumps(result), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--release-tag", default="")
    parser.add_argument("--asset-size-mib", type=int, default=1900)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--work-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    prepare_dataset(**vars(args))


if __name__ == "__main__":
    main()
