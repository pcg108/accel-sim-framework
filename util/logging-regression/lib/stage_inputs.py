#!/usr/bin/env python3
"""Stage an immutable Accel-Sim trace selection and hash its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


TRACE_RE = re.compile(r"^kernel-(\d+)\.(?:traceg|tracez)(?:\.xz)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ids(raw_ids: str) -> set[int] | None:
    if raw_ids == "all":
        return None
    try:
        result = {int(part) for part in raw_ids.split(",")}
    except ValueError as error:
        raise ValueError(f"invalid kernel ID list: {raw_ids}") from error
    if not result or any(kernel_id <= 0 for kernel_id in result):
        raise ValueError("kernel IDs must be positive integers")
    return result


def trace_reference(line: str) -> tuple[int, Path] | None:
    stripped = line.strip()
    candidate = Path(stripped)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe trace path in kernelslist.g: {stripped}")
    match = TRACE_RE.fullmatch(candidate.name)
    if match is None:
        return None
    return int(match.group(1)), candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--kernel-ids", required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve(strict=True)
    source_list = source_dir / "kernelslist.g"
    if not source_list.is_file():
        parser.error(f"missing kernels list: {source_list}")
    requested = parse_ids(args.kernel_ids)

    raw_text = source_list.read_text(encoding="utf-8")
    lines = raw_text.splitlines(keepends=True)
    seen: dict[int, Path] = {}
    parsed_lines: list[tuple[str, tuple[int, Path] | None]] = []
    for line in lines:
        reference = trace_reference(line)
        if reference is not None:
            kernel_id, relative_path = reference
            if kernel_id in seen:
                raise ValueError(f"duplicate kernel ID {kernel_id} in {source_list}")
            seen[kernel_id] = relative_path
        parsed_lines.append((line, reference))

    if not seen:
        raise ValueError(f"no kernel trace entries found in {source_list}")
    missing = set() if requested is None else requested.difference(seen)
    if missing:
        raise ValueError(
            "requested kernel IDs missing from kernelslist.g: "
            + ",".join(str(kernel_id) for kernel_id in sorted(missing))
        )
    selected_ids = set(seen) if requested is None else requested

    destination = args.destination
    destination.mkdir(parents=True, exist_ok=False)
    selected_lines: list[str] = []
    if requested is None:
        selected_lines = lines
    else:
        # Associate non-kernel commands (for example MemcpyHtoD) with the
        # following kernel, retaining them only when that kernel is selected.
        pending: list[str] = []
        for line, reference in parsed_lines:
            if reference is None:
                pending.append(line)
                continue
            kernel_id, _ = reference
            if kernel_id in selected_ids:
                selected_lines.extend(pending)
                selected_lines.append(line)
            pending.clear()

    staged_list = destination / "kernelslist.g"
    staged_list.write_text("".join(selected_lines), encoding="utf-8")

    files: list[dict[str, object]] = []
    checksum_rows: list[tuple[str, Path]] = [
        (sha256_file(source_list), source_list),
        (sha256_file(staged_list), staged_list),
    ]
    for kernel_id in sorted(selected_ids):
        relative_path = seen[kernel_id]
        source_path = (source_dir / relative_path).resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"trace is not a regular file: {source_path}")
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source_path, destination_path)
        digest = sha256_file(source_path)
        files.append(
            {
                "kernel_id": kernel_id,
                "relative_path": str(relative_path),
                "source_path": str(source_path),
                "sha256": digest,
                "size_bytes": source_path.stat().st_size,
            }
        )
        checksum_rows.append((digest, source_path))

    selection = {
        "source_directory": str(source_dir),
        "source_kernelslist": str(source_list),
        "source_kernelslist_sha256": checksum_rows[0][0],
        "staged_kernelslist": str(staged_list.resolve()),
        "staged_kernelslist_sha256": checksum_rows[1][0],
        "selected_kernel_ids": sorted(selected_ids),
        "trace_files": files,
    }
    args.selection_json.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.checksums.open("w", encoding="utf-8") as stream:
        for digest, path in checksum_rows:
            stream.write(f"{digest}  {path}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
