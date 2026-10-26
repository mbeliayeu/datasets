"""Offline tests of snapshot completeness and release archive reconstruction."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mirror_hf_dataset import ArchiveParts, make_archive, prepare_dataset


class ArchiveTests(unittest.TestCase):
    def test_multiple_subsets_splits_and_hidden_files_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            output = root / "assets"
            output.mkdir()
            expected = {
                ".gitattributes": b"*.parquet filter=lfs\n",
                "README.md": b"license: cc-by-4.0\n",
                "corpus/train-00000.parquet": bytes(range(256)) * 10,
                "corpus/test-00000.parquet": os.urandom(2048),
                "queries/validation-00000.parquet": b"all queries",
                "documents_metadata/test.parquet": b"documents",
                "qrels/test.parquet": b"relevance",
            }
            for name, data in expected.items():
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            cache = snapshot / ".cache/huggingface/download/internal.lock"
            cache.parent.mkdir(parents=True)
            cache.write_text("local metadata")
            files, parts = make_archive(snapshot, list(expected), output, "dataset", 256)
            self.assertGreater(len(parts), 1)
            self.assertEqual({f["path"] for f in files}, set(expected))
            reconstructed = bytearray()
            for part in parts:
                data = (output / part["name"]).read_bytes()
                self.assertLessEqual(len(data), 256)
                self.assertEqual(len(data), part["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), part["sha256"])
                reconstructed.extend(data)
            with tarfile.open(fileobj=io.BytesIO(reconstructed), mode="r:gz") as archive:
                actual = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
            self.assertEqual(actual, {f"dataset/{name}": data for name, data in expected.items()})

    def test_single_archive_has_tar_gz_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("dataset")
            _, parts = make_archive(root, ["README.md"], root, "dataset", 1024**2)
            self.assertEqual(parts[0]["name"], "dataset.tar.gz")
            with tarfile.open(root / "dataset.tar.gz") as archive:
                self.assertEqual(archive.extractfile("dataset/README.md").read(), b"dataset")

    def test_exact_boundary_does_not_create_empty_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArchiveParts(Path(temporary), "data", 4)
            writer.write(b"abcdefgh")
            writer.close()
            self.assertEqual([p["bytes"] for p in writer.parts], [4, 4])
            self.assertEqual(len(list(Path(temporary).iterdir())), 2)

    def test_asset_count_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArchiveParts(Path(temporary), "data", 4, max_parts=2)
            with self.assertRaisesRegex(ValueError, "Too many"):
                writer.write(b"123456789")
            writer.close()

    def test_rejects_traversal_missing_files_and_symlinks_outside_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (root / "secret").write_text("must not be archived")
            (snapshot / "link").symlink_to(root / "secret")
            for name in ["../secret", "/etc/passwd", "missing.parquet", "link"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    make_archive(snapshot, [name], root, "dataset", 1024**2)


class SnapshotTests(unittest.TestCase):
    def test_download_is_pinned_unfiltered_and_manifest_matches(self):
        sha = "a" * 40
        info = SimpleNamespace(sha=sha, siblings=[
            SimpleNamespace(rfilename="corpus/test.parquet", size=3),
            SimpleNamespace(rfilename="queries/test.parquet", size=5),
        ])
        api = Mock()
        api.dataset_info.return_value = info

        def download(**kwargs):
            root = Path(kwargs["local_dir"])
            for name, data in [("corpus/test.parquet", b"abc"), ("queries/test.parquet", b"12345")]:
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(data)
            return str(root)

        downloader = Mock(side_effect=download)
        hub = SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=downloader)
        validators = SimpleNamespace(validate_repo_id=Mock())
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"huggingface_hub": hub, "huggingface_hub.utils": validators}
        ), patch.dict(os.environ, {"HF_TOKEN": "", "GITHUB_OUTPUT": "", "GITHUB_RUN_ID": "123"}):
            root = Path(temporary)
            manifest = prepare_dataset("owner/data", "moving-branch", root)
            call = downloader.call_args.kwargs
            self.assertEqual(call["revision"], sha)
            self.assertEqual(call["repo_type"], "dataset")
            self.assertNotIn("allow_patterns", call)
            self.assertNotIn("ignore_patterns", call)
            self.assertEqual(manifest["commit_sha"], sha)
            self.assertEqual(len(manifest["files"]), 2)
            assets = root / "assets"
            self.assertEqual(json.loads((assets / "manifest.json").read_text()), manifest)
            for line in (assets / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((assets / name).read_bytes()).hexdigest(), digest)
            self.assertFalse(list(root.glob("snapshot-*")))
            with self.assertRaisesRegex(ValueError, "must be empty"):
                prepare_dataset("owner/data", "main", root)


if __name__ == "__main__":
    unittest.main()
