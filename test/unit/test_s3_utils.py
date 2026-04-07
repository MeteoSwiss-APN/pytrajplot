"""Unit tests for pytrajplot.s3_utils."""
import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError

from pytrajplot.s3_utils import download_s3_prefix, upload_dir_to_s3


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "operation")


class TestDownloadS3Prefix:
    """Tests for download_s3_prefix."""

    def _make_paginator(self, pages: list[list[dict]]) -> MagicMock:
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": page} for page in pages
        ]
        return paginator

    def test_downloads_files_to_local_dir(self, tmp_path):
        s3 = MagicMock()
        s3.get_paginator.return_value = self._make_paginator(
            [[{"Key": "prefix/subdir/file.txt"}, {"Key": "prefix/other.txt"}]]
        )

        download_s3_prefix(s3, "my-bucket", "prefix/", str(tmp_path))

        assert s3.download_file.call_count == 2
        s3.download_file.assert_any_call(
            "my-bucket", "prefix/subdir/file.txt", str(tmp_path / "subdir" / "file.txt")
        )
        s3.download_file.assert_any_call(
            "my-bucket", "prefix/other.txt", str(tmp_path / "other.txt")
        )

    def test_skips_prefix_itself(self, tmp_path):
        """Objects whose key equals the prefix (no relative path) are skipped."""
        s3 = MagicMock()
        s3.get_paginator.return_value = self._make_paginator(
            [[{"Key": "prefix/"}, {"Key": "prefix/file.txt"}]]
        )

        download_s3_prefix(s3, "my-bucket", "prefix/", str(tmp_path))

        assert s3.download_file.call_count == 1

    def test_empty_prefix_raises_runtime_error(self, tmp_path):
        s3 = MagicMock()
        s3.get_paginator.return_value = self._make_paginator([[]])

        with pytest.raises(RuntimeError, match="No files found"):
            download_s3_prefix(s3, "my-bucket", "empty/", str(tmp_path))

    def test_no_such_bucket_raises_runtime_error(self, tmp_path):
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.side_effect = _client_error("NoSuchBucket")
        s3.get_paginator.return_value = paginator

        with pytest.raises(RuntimeError, match="does not exist"):
            download_s3_prefix(s3, "missing-bucket", "prefix/", str(tmp_path))

    def test_access_denied_raises_runtime_error(self, tmp_path):
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.side_effect = _client_error("AccessDenied")
        s3.get_paginator.return_value = paginator

        with pytest.raises(RuntimeError, match="Access denied"):
            download_s3_prefix(s3, "my-bucket", "prefix/", str(tmp_path))

class TestUploadDirToS3:
    """Tests for upload_dir_to_s3."""

    def test_uploads_all_files(self, tmp_path):
        (tmp_path / "file1.pdf").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file2.png").write_text("b")

        s3 = MagicMock()
        upload_dir_to_s3(s3, str(tmp_path), "my-bucket", "output/")

        assert s3.upload_file.call_count == 2
        uploaded_keys = {c.args[2] for c in s3.upload_file.call_args_list}
        assert "output/file1.pdf" in uploaded_keys
        assert "output/sub/file2.png" in uploaded_keys

    def test_metadata_passed_as_extra_args(self, tmp_path):
        (tmp_path / "file.pdf").write_text("x")
        s3 = MagicMock()
        metadata = {"product_publisher": "forecast-iconch1eps-trajectories"}

        upload_dir_to_s3(s3, str(tmp_path), "my-bucket", "output/", metadata=metadata)

        call_kwargs = s3.upload_file.call_args.kwargs
        assert call_kwargs.get("ExtraArgs") == {"Metadata": metadata}

    def test_no_metadata_passes_empty_extra_args(self, tmp_path):
        (tmp_path / "file.pdf").write_text("x")
        s3 = MagicMock()

        upload_dir_to_s3(s3, str(tmp_path), "my-bucket", "output/")

        call_kwargs = s3.upload_file.call_args.kwargs
        assert call_kwargs.get("ExtraArgs") == {}

    def test_empty_directory_raises_runtime_error(self, tmp_path):
        s3 = MagicMock()

        with pytest.raises(RuntimeError, match="No output files"):
            upload_dir_to_s3(s3, str(tmp_path), "my-bucket", "output/")

    def test_no_such_bucket_raises_runtime_error(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        s3 = MagicMock()
        s3.upload_file.side_effect = _client_error("NoSuchBucket")

        with pytest.raises(RuntimeError, match="does not exist"):
            upload_dir_to_s3(s3, str(tmp_path), "missing-bucket", "prefix/")

    def test_access_denied_raises_runtime_error(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        s3 = MagicMock()
        s3.upload_file.side_effect = _client_error("AccessDenied")

        with pytest.raises(RuntimeError, match="Access denied"):
            upload_dir_to_s3(s3, str(tmp_path), "my-bucket", "prefix/")
