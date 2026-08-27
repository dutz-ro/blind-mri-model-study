from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..errors import MRIStudyError
from ..utils import atomic_json


def validate_tree(root: Path, report_path: Path) -> dict:
    dcmdump = shutil.which("dcmdump")
    if not dcmdump:
        raise MRIStudyError("DCMTK dcmdump is not installed.")
    checked = 0
    failures = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        result = subprocess.run(
            [dcmdump, "-q", "-M", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        checked += 1
        if result.returncode:
            failures.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "returncode": result.returncode,
                    "error_class": "dcmdump-validation-error",
                }
            )
    report = {
        "validator": "dcmtk-dcmdump",
        "checked": checked,
        "failures": failures,
        "passed": not failures,
    }
    atomic_json(report_path, report)
    return report
