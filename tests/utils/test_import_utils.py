#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import sys
from pathlib import Path

from pytest_mock import MockerFixture

from utils import import_utils
from utils.import_utils import import_modules_from_folder


def test_import_utils(tmp_path: Path, mocker: MockerFixture) -> None:
    tmp_path_str = str(tmp_path)
    sys.path.append(tmp_path_str)
    mocker.patch.object(import_utils, "LIBRARY_ROOT", tmp_path)
    try:
        files = [
            "my_test_parent/child/module.py",
            "my_test_parent/child/nested/module.py",
            "my_test_parent/sibling.py",
            "my_internal/my_test_parent/child/module.py",
            "my_internal/my_test_parent/sibling.py",
            "my_internal/projects/A/my_test_parent/child/module.py",
            "my_internal/projects/B/my_test_parent/child/module.py",
        ]
        for path in files:
            path = tmp_path / path
            for package in path.parents:
                if package == tmp_path:
                    break
                package.mkdir(exist_ok=True, parents=True)
                if not package.joinpath("__init__.py").exists():
                    package.joinpath("__init__.py").write_bytes(b"")
            path.write_bytes(b"")

        import_modules_from_folder(
            "my_test_parent/child",
            extra_roots=["my_internal", "my_internal/projects/*"],
        )
        assert "my_test_parent.child.module" in sys.modules
        assert "my_test_parent.child.nested.module" in sys.modules
        assert "my_test_parent.sibling" not in sys.modules
        assert "my_internal.my_test_parent.child.module" in sys.modules
        assert "my_internal.my_test_parent.sibling" not in sys.modules
        assert "my_internal.projects.A.my_test_parent.child.module" in sys.modules
        assert "my_internal.projects.B.my_test_parent.child.module" in sys.modules
    finally:
        sys.path.remove(tmp_path_str)
