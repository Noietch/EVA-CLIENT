"""Vendored utility subset from eva/engine/utils — only the leaves needed by
the vendored config + registry. Trimmed of distributed, training, and CLI helpers.
"""

from __future__ import annotations

import importlib
import importlib.util
import os.path as osp
import warnings
from collections import abc
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from inspect import ismodule
from typing import Any

from packaging.version import parse


def is_str(x: Any) -> bool:
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(
    imports: str | list[str] | None, allow_failed_imports: bool = False
):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning, stacklevel=2)
                imported_tmp = None
            else:
                raise ImportError(f"Failed to import {imp}") from None
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def is_seq_of(seq: Any, expected_type: type | tuple, seq_type: type | None = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from eva.engine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq: Any, expected_type: type | tuple[type, ...]) -> bool:
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def get_object_from_string(obj_name: str):
    """Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    # Split the dotted path into an iterator so we can walk it element by element
    parts = iter(obj_name.split("."))
    module_name = next(parts)
    module: Any = None
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            # mmcv.ops has nms.py and nms function at the same time. So the
            # function will have a higher priority
            obj = getattr(module, part, None)
            if obj is not None and not ismodule(obj):
                break
            module_name = f"{module_name}.{part}"
        except StopIteration:
            # if obj is a module
            return module
        except ImportError:
            return None

    obj = module
    # Continue walking the remaining dotted parts as attribute accesses
    while True:
        try:
            obj = getattr(obj, part)
            part = next(parts)
        except StopIteration:
            return obj
        except AttributeError:
            return None


def check_file_exist(filename: str, msg_tmpl: str = 'file "{}" does not exist') -> None:
    """Raise ``FileNotFoundError`` if ``filename`` is not an existing regular file.

    Args:
        filename: Path to the file to be checked.
        msg_tmpl: Template for the error message; ``{}`` is replaced with ``filename``.
    """
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert "parrots" not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version_str}"
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {"a": -3, "b": -2, "rc": -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(
                    f"unknown prerelease version {version.pre[0]}, "
                    "version checking may go wrong",
                    stacklevel=2,
                )
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # First check if it's an importable module
    spec = importlib.util.find_spec(package)
    if spec is not None and spec.origin is not None:
        return True

    # If not found as module, check if it's a distribution package
    try:
        distribution(package)
        return True
    except PackageNotFoundError:
        return False


def package2module(package: str) -> str:
    """Infer the top-level import module name of an installed distribution.

    Args:
        package (str): Name of the distribution package.

    Returns:
        str: The top-level module name imported from the package.
    """
    pkg = distribution(package)
    top_level = pkg.read_text("top_level.txt")
    if top_level:
        return top_level.split("\n")[0]
    for path in pkg.files or []:
        if path.name == "__init__.py" and len(path.parts) == 2:
            return path.parts[0]
    raise ValueError(f"can not infer the module name of {package}")


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    # Try to get location from distribution package metadata
    location = None
    try:
        dist = distribution(package)
        locate_result: Any = dist.locate_file("")
        location = str(locate_result.parent)
    except PackageNotFoundError:
        pass

    # If distribution package not found, try to find via importlib
    if location is None:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            if spec.origin is not None:
                return osp.dirname(spec.origin)
            else:
                # `get_installed_path` cannot get the installed path of
                # namespace packages
                raise RuntimeError(
                    f"{package} is a namespace package, "
                    "which is invalid for `get_install_path`"
                )
        else:
            raise PackageNotFoundError(f"Package {package} is not installed")

    # Check if package directory exists in the location
    possible_path = osp.join(location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(location, package2module(package))


