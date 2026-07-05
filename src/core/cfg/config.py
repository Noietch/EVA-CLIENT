"""Vendored mmengine-style Config + ConfigDict, with LazyObject/LazyAttr,
read_base context manager, the AST helpers that parse lazy .py configs, and a
small yaml/json io shim used by Config.fromfile + Config.dump.

Merged from upstream engine/config/{config,lazy,utils}.py and fileio yaml/json
handlers. Pruned of model-index / external-package config helpers and the
yapf/rich-dependent pretty-print + diff utilities.
"""

from __future__ import annotations

import ast
import copy
import difflib
import importlib
import importlib.util
import json
import logging
import os
import os.path as osp
import platform
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from argparse import Action, ArgumentParser, Namespace
from collections import OrderedDict, abc, defaultdict
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, NoReturn, SupportsIndex

import yaml
from addict import Dict
from rich.console import Console
from rich.text import Text

from ._utils import (
    check_file_exist,
    digit_version,
    get_installed_path,
    import_modules_from_strings,
    is_installed,
    is_seq_of,
)

try:
    from yaml import CDumper as _YamlDumper  # type: ignore
    from yaml import CLoader as _YamlLoader  # type: ignore
except ImportError:
    from yaml import Dumper as _YamlDumper  # type: ignore
    from yaml import Loader as _YamlLoader  # type: ignore

logger = logging.getLogger(__name__)

# ===== from lazy.py =====

class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
        location (str, optional): The filename and line number of the imported
            module statement happened.
    """

    def __init__(
        self,
        module: str | list[str] | tuple[str, ...],
        imported: str | None = None,
        location: str | None = None,
    ) -> None:
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError(
                "module should be `str`, `list`, or `tuple`"
                f"but got {type(module)}, this might be "
                "a bug of engine, please report it to "
                "https://github.com/open-mmlab/engine/issues"
            )
        self._module: str | list[str] | tuple[str, ...] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError(
                "imported should be `str` or None, but got "
                f"{type(imported)}, this might be "
                "a bug of engine, please report it to "
                "https://github.com/open-mmlab/engine/issues"
            )
        self._imported = imported
        self.location = location

    def build(self) -> object:
        """Actually import and return the module (or imported attribute) this LazyObject refers to."""  # noqa: E501
        if isinstance(self._module, str):
            # Single-module case: import it directly
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f"Failed to import {self._module} in {self.location} for {e}") from e

            # Dereference an explicitly imported symbol (from module import name)
            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f"Failed to import {self._imported} from {self._module} in {self.location}"
                    )

            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            try:
                # Import every submodule so attribute access works, then return the top package
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split(".")[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f"Failed to import {self.module} in {self.location} for {e}") from e

    @property
    def module(self) -> str:
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split(".")[0]

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError()

    def __deepcopy__(self, memo: dict[int, object]) -> LazyObject:
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name: str) -> LazyAttr:
        if self.location is not None:
            location = self.location.split(", line")[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyObject will return a `__getstate__` LazyObject` or `__setstate__`
    # LazyObject.
    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state


class LazyAttr:
    """The attribute of the LazyObject.

    When parsing the configuration file, the imported syntax will be
    parsed as the assignment ``LazyObject``. During the subsequent parsing
    process, users may reference the attributes of the LazyObject.
    To ensure that these attributes also contain information needed to
    reconstruct the attribute itself, LazyAttr was introduced.

    Examples:
        >>> models = LazyObject(['mmdet.models'])
        >>> model = dict(type=models.RetinaNet)
        >>> print(type(model['type']))  # <class 'engine.config.lazy.LazyAttr'>
        >>> print(model['type'].build())  # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
    """  # noqa: E501

    def __init__(
        self, name: str, source: LazyObject | LazyAttr, location: str | None = None
    ) -> None:
        self.name = name
        self.source: LazyAttr | LazyObject = source

        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                if self.source._imported is None:
                    # source code:
                    # from xxx.yyy import zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy', 'zzz')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy.zzz"
                    self._module = self.source._module
                else:
                    # source code:
                    # import xxx.yyy as zzz
                    # equivalent code:
                    # zzz = LazyObject('xxx.yyy')
                    # The source code of get attribute:
                    # eee = zzz.eee
                    # Then, `eee._module` should be "xxx.yyy"
                    self._module = f"{self.source._module}.{self.source}"
            else:
                # The source code of LazyObject should be
                # 1. import xxx.yyy
                # 2. import xxx.zzz
                # Equivalent to
                # xxx = LazyObject(['xxx.yyy', 'xxx.zzz'])

                # The source code of LazyAttr should be
                # eee = xxx.eee
                # Then, eee._module = xxx
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            # 1. import xxx
            # 2. zzz = xxx.yyy.zzz

            # Equivalent to:
            # xxx = LazyObject('xxx')
            # zzz = xxx.yyy.zzz
            # zzz._module = xxx.yyy._module + zzz.name
            self._module = f"{self.source._module}.{self.source.name}"
        self.location = location

    @property
    def module(self) -> str:
        return self._module

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError()

    def __getattr__(self, name: str) -> LazyAttr:
        return LazyAttr(name, self)

    def __deepcopy__(self, memo: dict[int, object]) -> LazyAttr:
        return LazyAttr(self.name, self.source)

    def build(self) -> object:
        """Build the source LazyObject/LazyAttr and return ``self.name`` attribute from it."""
        obj = self.source.build()
        try:
            return getattr(obj, self.name)
        except AttributeError as e:
            raise ImportError(
                f"Failed to import {self.module}.{self.name} in {self.location}"
            ) from e
        except ImportError:
            raise

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

# ===== from fileio_shim.py =====

try:
    from yaml import CDumper as _YamlDumper  # type: ignore
    from yaml import CLoader as _YamlLoader  # type: ignore
except ImportError:
    from yaml import Dumper as _YamlDumper  # type: ignore
    from yaml import Loader as _YamlLoader  # type: ignore


def _infer_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lstrip(".").lower()
    if suffix in ("yaml", "yml"):
        return "yaml"
    if suffix == "json":
        return "json"
    raise ValueError(f"Unsupported file format for {path!r}; expected .yaml/.yml/.json")


def load(file: str | Path, file_format: str | None = None) -> Any:
    """Load yaml/json from ``file`` (path). file_format auto-inferred from suffix."""
    fmt = (file_format or _infer_format(file)).lower()
    with open(file, encoding="utf-8") as f:
        if fmt in ("yaml", "yml"):
            return yaml.load(f, Loader=_YamlLoader)
        return json.load(f)


def dump(obj: Any, file: str | Path | None = None, file_format: str | None = None) -> str | None:
    """Dump ``obj`` to ``file`` (path) or return a string if file is None."""
    fmt = file_format
    if fmt is None and file is not None:
        fmt = _infer_format(file)
    if fmt is None:
        raise ValueError("Either file or file_format must be provided")
    fmt = fmt.lower()

    if file is None:
        if fmt in ("yaml", "yml"):
            return yaml.dump(obj, Dumper=_YamlDumper)
        return json.dumps(obj, default=str)

    with open(file, "w", encoding="utf-8") as f:
        if fmt in ("yaml", "yml"):
            yaml.dump(obj, f, Dumper=_YamlDumper)
        else:
            json.dump(obj, f, default=str)
    return None

# ===== from config_utils.py =====

PYTHON_ROOT_DIR = osp.dirname(osp.dirname(sys.executable))
SYSTEM_PYTHON_PREFIX = "/usr/lib/python"

MODULE2PACKAGE: dict[str, str] = {}


class ConfigParsingError(RuntimeError):
    """Raised when the lazy-import config parser encounters invalid syntax or unresolvable imports."""  # noqa: E501


def _get_cfg_metainfo(package_path: str, cfg_path: str) -> dict:
    """Get target meta information from all 'metafile.yml' defined in `mode-
    index.yml` of external package.

    Args:
        package_path (str): Path of external package.
        cfg_path (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, ".mim", "model-index.yml")
    meta_index = load(meta_index_path)
    cfg_dict = dict()
    for meta_path in meta_index["Import"]:
        meta_path = osp.join(package_path, ".mim", meta_path)
        cfg_meta = load(meta_path)
        for model_cfg in cfg_meta["Models"]:
            if "Config" not in model_cfg:
                warnings.warn(f"There is not `Config` define in {model_cfg}", stacklevel=2)
                continue
            cfg_name = model_cfg["Config"].partition("/")[-1]
            # Some config could have multiple weights, we only pick the
            # first one.
            if cfg_name in cfg_dict:
                continue
            cfg_dict[cfg_name] = model_cfg
    if cfg_path not in cfg_dict:
        raise ValueError(f"Expected configs: {cfg_dict.keys()}, but got {cfg_path}")
    return cfg_dict[cfg_path]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    """Get config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_file = cfg_file.split(".")[0]
    model_cfg = _get_cfg_metainfo(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg["Config"])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str) -> str:
    """Get base config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, ".mim", "configs", cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def _get_package_and_cfg_path(cfg_path: str) -> tuple[str, str]:
    """Get package name and relative config path.

    Args:
        cfg_path (str): External relative config path with 'package::'.

    Returns:
        tuple[str, str]: Package name and config path.
    """
    if re.match(r"\w*::\w*/\w*", cfg_path) is None:
        raise ValueError(
            "`_get_package_and_cfg_path` is used for get external package, "
            "please specify the package name and relative config path, just "
            "like `mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`"
        )
    package_cfg = cfg_path.split("::")
    if len(package_cfg) > 2:
        raise ValueError(
            f"`::` should only be used to separate package and config name, "
            f"but found multiple `::` in {cfg_path}"
        )
    package, cfg_path = package_cfg
    assert package in MODULE2PACKAGE, f"engine does not support to load {package} config."
    package = MODULE2PACKAGE[package]
    return package, cfg_path


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.key:
            return None
        else:
            return node


def _is_builtin_module(module_name: str) -> bool:
    """Check if a module is a built-in module.

    Arg:
        module_name: name of module.
    """
    if module_name.startswith("."):
        return False
    if module_name.startswith("engine.config"):
        return True
    if module_name.startswith("configs."):
        return True
    if module_name in sys.stdlib_module_names:
        return True
    spec = find_spec(module_name.split(".")[0])
    # Module not found
    if spec is None:
        return False
    origin_path = getattr(spec, "origin", None)
    if origin_path is None:
        return False
    if origin_path == "frozen":
        return True
    origin_path = osp.abspath(origin_path)
    if (
        "site-package" in origin_path
        or "dist-package" in origin_path
        or not origin_path.startswith((PYTHON_ROOT_DIR, SYSTEM_PYTHON_PREFIX))
    ):
        return False
    else:
        return True


class ImportTransformer(ast.NodeTransformer):
    """Convert the import syntax to the assignment of
    :class:`engine.config.LazyObject` and preload the base variable before
    parsing the configuration file.

    Since you are already looking at this part of the code, I believe you must
    be interested in the mechanism of the ``lazy_import`` feature of
    :class:`Config`. In this docstring, we will dive deeper into its
    principles.

    Most of OpenMMLab users maybe bothered with that:

        * In most of popular IDEs, they cannot navigate to the source code in
          configuration file
        * In most of popular IDEs, they cannot jump to the base file in current
          configuration file, which is much painful when the inheritance
          relationship is complex.

    In order to solve this problem, we introduce the ``lazy_import`` mode.

    A very intuitive idea for solving this problem is to import the module
    corresponding to the "type" field using the ``import`` syntax. Similarly,
    we can also ``import`` base file.

    However, this approach has a significant drawback. It requires triggering
    the import logic to parse the configuration file, which can be
    time-consuming. Additionally, it implies downloading numerous dependencies
    solely for the purpose of parsing the configuration file.
    However, it's possible that only a portion of the config will actually be
    used. For instance, the package used in the ``train_pipeline`` may not
    be necessary for an evaluation task. Forcing users to download these
    unused packages is not a desirable solution.

    To avoid this problem, we introduce :class:`engine.config.LazyObject` and
    :class:`engine.config.LazyAttr`. Before we proceed with further
    explanations, you may refer to the documentation of these two modules to
    gain an understanding of their functionalities.

    Actually, one of the functions of ``ImportTransformer`` is to hack the
    ``import`` syntax. It will replace the import syntax
    (exclude import the base files) with the assignment of ``LazyObject``.

    As for the import syntax of the base file, we cannot lazy import it since
    we're eager to merge the fields of current file and base files. Therefore,
    another function of the ``ImportTransformer`` is to collaborate with
    ``Config._parse_lazy_import`` to parse the base files.

    Args:
        global_dict (dict): The global dict of the current configuration file.
            If we divide ordinary Python syntax into two parts, namely the
            import section and the non-import section (assuming a simple case
            with imports at the beginning and the rest of the code following),
            the variables generated by the import statements are stored in
            global variables for subsequent code use. In this context,
            the ``global_dict`` represents the global variables required when
            executing the non-import code. ``global_dict`` will be filled
            during visiting the parsed code.
        base_dict (dict): All variables defined in base files.

            Examples:
                >>> from eva.engine.config import read_base
                >>>
                >>>
                >>> with read_base():
                >>>     from .._base_.default_runtime import *
                >>>     from .._base_.datasets.coco_detection import dataset

            In this case, the base_dict will be:

            Examples:
                >>> base_dict = {
                >>>     '.._base_.default_runtime': ...
                >>>     '.._base_.datasets.coco_detection': dataset}

            and `global_dict` will be updated like this:

            Examples:
                >>> global_dict.update(base_dict['.._base_.default_runtime'])  # `import *` means update all data
                >>> global_dict.update(dataset=base_dict['.._base_.datasets.coco_detection']['dataset'])  # only update `dataset`
    """  # noqa: E501

    def __init__(
        self,
        global_dict: dict,
        base_dict: dict | None = None,
        filename: str | None = None,
    ) -> None:
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        # In Windows, the filename could be like this:
        # "C:\\Users\\runneradmin\\AppData\\Local\\"
        # Although it has been an raw string, ast.parse will firstly escape
        # it as the executed code:
        # "C:\Users\runneradmin\AppData\Local\\\"
        # As you see, the `\U` will be treated as a part of
        # the escape sequence during code parsing, leading to an
        # parsing error
        # Here we use `encode('unicode_escape').decode()` for double escaping
        if isinstance(filename, str):
            filename = filename.encode("unicode_escape").decode()
        self.filename = filename
        self.imported_obj: set = set()
        super().__init__()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.Assign] | ast.ImportFrom | None:
        """Hack the ``from ... import ...`` syntax and update the global_dict.

        Examples:
            >>> from mmdet.models import RetinaNet

        Will be parsed as:

        Examples:
            >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')

        ``global_dict`` will also be updated by ``base_dict`` as the
        class docstring says.

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            list[ast.Assign] | None: There three cases:

                * If the node is a statement of importing base files.
                  None will be returned.
                * If the node is a statement of importing a builtin module,
                  node will be directly returned
                * Otherwise, it will return the assignment statements of
                  ``LazyObject``.
        """
        # Built-in modules will not be parsed as LazyObject
        module = f"{node.level * '.'}{node.module}"
        if _is_builtin_module(module):
            # Make sure builtin module will be added into `self.imported_obj`
            for alias in node.names:
                if alias.asname is not None:
                    self.imported_obj.add(alias.asname)
                elif alias.name == "*":
                    raise ConfigParsingError("Cannot import * from non-base config")
                else:
                    self.imported_obj.add(alias.name)
            return node

        if module in self.base_dict:
            for alias_node in node.names:
                if alias_node.name == "*":
                    self.global_dict.update(self.base_dict[module])
                    return None
                if alias_node.asname is not None:
                    base_key = alias_node.asname
                else:
                    base_key = alias_node.name
                self.global_dict[base_key] = self.base_dict[module][alias_node.name]
            return None

        nodes: list[ast.Assign] = []
        for alias_node in node.names:
            # `ast.alias` has lineno attr after Python 3.10,
            if hasattr(alias_node, "lineno"):
                lineno = alias_node.lineno
            else:
                lineno = node.lineno
            if alias_node.name == "*":
                # TODO: If users import * from a non-config module, it should
                # fallback to import the real module and raise a warning to
                # remind users the real module will be imported which will slow
                # down the parsing speed.
                raise ConfigParsingError(
                    "Illegal syntax in config! `from xxx import *` is not "
                    "allowed to appear outside the `if base:` statement"
                )
            elif alias_node.asname is not None:
                # case1:
                # from eva.engine.dataset import BaseDataset as Dataset ->
                # Dataset = LazyObject('engine.dataset', 'BaseDataset')
                code = f'{alias_node.asname} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.asname)
            else:
                # case2:
                # from eva.engine.model import BaseModel
                # BaseModel = LazyObject('engine.model', 'BaseModel')
                code = (
                    f'{alias_node.name} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                )
                self.imported_obj.add(alias_node.name)
            try:
                nodes.append(ast.parse(code).body[0])  # type: ignore
            except Exception as e:
                raise ConfigParsingError(
                    f"Cannot import {alias_node} from {module}"
                    "1. Cannot import * from 3rd party lib in the config "
                    "file\n"
                    "2. Please check if the module is a base config which "
                    "should be added to `_base_`\n"
                ) from e
        return nodes

    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Work with ``_gather_abs_import_lazyobj`` to hack the ``import ...``
        syntax.

        Examples:
            >>> import mmcls.models
            >>> import mmcls.datasets
            >>> import mmcls

        Will be parsed as:

        Examples:
            >>> # import mmcls.models; import mmcls.datasets; import mmcls
            >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            ast.Assign: If the import statement is ``import ... as ...``,
            ast.Assign will be returned, otherwise node will be directly
            returned.
        """
        # For absolute import like: `import mmdet.configs as configs`.
        # It will be parsed as:
        # configs = LazyObject('mmdet.configs')
        # For absolute import like:
        # `import mmdet.configs`
        # `import mmdet.configs.default_runtime`
        # This will be parsed as
        # mmdet = LazyObject(['mmdet.configs.default_runtime', 'mmdet.configs])
        # However, visit_Import cannot gather other import information, so
        # `_gather_abs_import_LazyObject` will gather all import information
        # from the same module and construct the LazyObject.
        alias_list = node.names
        assert len(alias_list) == 1, (
            "Illegal syntax in config! import multiple modules in one line is not supported"
        )
        # TODO Support multiline import
        alias = alias_list[0]
        if alias.asname is not None:
            self.imported_obj.add(alias.asname)
            if _is_builtin_module(alias.name.split(".")[0]):
                return node
            return ast.parse(  # type: ignore
                f'{alias.asname} = LazyObject("{alias.name}",'
                f'location="{self.filename}, line {node.lineno}")'
            ).body[0]
        return node


def _gather_abs_import_lazyobj(tree: ast.Module, filename: str | None = None):
    """Coalesce bare ``import x.y`` statements into single LazyObject assignments per top-level module."""  # noqa: E501
    if isinstance(filename, str):
        filename = filename.encode("unicode_escape").decode()
    imported = defaultdict(list)
    abs_imported = set()
    new_body: list[ast.stmt] = []
    # module2node is used to get lineno when Python < 3.10
    module2node: dict = dict()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip converting built-in module to LazyObject
                if _is_builtin_module(alias.name):
                    new_body.append(node)
                    continue
                module = alias.name.split(".")[0]
                module2node.setdefault(module, node)
                imported[module].append(alias)
            continue
        new_body.append(node)

    for key, value in imported.items():
        names = [_value.name for _value in value]
        if hasattr(value[0], "lineno"):
            lineno = value[0].lineno
        else:
            lineno = module2node[key].lineno
        lazy_module_assign = ast.parse(
            f'{key} = LazyObject({names}, location="{filename}, line {lineno}")'  # noqa: E501
        )  # noqa: E501
        abs_imported.add(key)
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree, abs_imported

# ===== Config + ConfigDict =====

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]

# Silence noisy pydantic field attribute warnings (e.g., repr/frozen on Field)
# without importing pydantic internals; they do not affect runtime behavior.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\._internal\._generate_schema",
)

if platform.system() == "Windows":
    import regex as re  # type: ignore[import-not-found]
else:
    import re  # type: ignore


def _lazy2string(cfg_dict: dict[str, Any], dict_type: type[Any] | None = None):
    if isinstance(cfg_dict, dict):
        dict_type = dict_type or type(cfg_dict)
        return dict_type({k: _lazy2string(v, dict_type) for k, v in dict.items(cfg_dict)})
    elif isinstance(cfg_dict, (tuple, list)):
        return type(cfg_dict)(_lazy2string(v, dict_type) for v in cfg_dict)
    elif isinstance(cfg_dict, (LazyAttr, LazyObject)):
        return f"{cfg_dict.module}.{str(cfg_dict)}"
    else:
        return cfg_dict


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.

    If the class attribute ``lazy``  is ``False``, users will get the
    object built by ``LazyObject`` or ``LazyAttr``, otherwise users will get
    the ``LazyObject`` or ``LazyAttr`` itself.

    The ``lazy`` should be set to ``True`` to avoid building the imported
    object during configuration parsing, and it should be set to False outside
    the Config to ensure that users do not experience the ``LazyObject``.
    """

    lazy = False

    def __init__(__self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(__self, "__parent", kwargs.pop("__parent", None))
        object.__setattr__(__self, "__key", kwargs.pop("__key", None))
        object.__setattr__(__self, "__frozen", False)
        for arg in args:
            if not arg:
                continue
            # Since ConfigDict.items will convert LazyObject to real object
            # automatically, we need to call super().items() to make sure
            # the LazyObject will not be converted.
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and len(arg) > 0 and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):  # pyright: ignore[reportGeneralTypeIssues]
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    def __missing__(self, name: str) -> Any:
        raise KeyError(name)

    def __getattr__(self, name: str) -> Any:
        try:
            value = super().__getattr__(name)
            if isinstance(value, (LazyAttr, LazyObject)) and not self.lazy:
                value = value.build()
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e
        except Exception:
            raise
        else:
            return value

    @classmethod
    def _hook(cls, item: Any):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __setattr__(self, name: str, value: Any) -> None:
        value = self._hook(value)
        return super().__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        value = self._hook(value)
        return super().__setitem__(name, value)

    def __getitem__(self, key: str):
        return self.build_lazy(super().__getitem__(key))

    def __deepcopy__(self, memo: dict[int, object]):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def __copy__(self):
        other = self.__class__()
        for key, value in super().items():
            other[key] = value
        return other

    copy = __copy__

    def __iter__(self) -> Iterator[Any]:
        # Implement `__iter__` to overwrite the unpacking operator `**cfg_dict`
        # to get the built lazy object
        return iter(self.keys())

    def get(self, key: str, default: Any | None = None) -> Any:
        """Get the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().get(key, default))

    def pop(self, key: str, default: Any | None = None):
        """Pop the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().pop(key, default))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Override this method to make sure the LazyObject will not be built
        during updating."""
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError("update only accept one positional argument")
            # Avoid to used self.items to build LazyObject
            for key, value in dict.items(args[0]):  # pyright: ignore[reportGeneralTypeIssues]
                other[key] = value

        for key, value in dict(kwargs).items():
            other[key] = value
        for k, v in other.items():
            if (k not in self) or (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
                self[k] = self._hook(v)
            else:
                self[k].update(v)

    def build_lazy(self, value: Any) -> Any:
        """If class attribute ``lazy`` is False, the LazyObject will be built
        and returned.

        Args:
            value (Any): The value to be built.

        Returns:
            Any: The built value.
        """
        if isinstance(value, (LazyAttr, LazyObject)) and not self.lazy:
            value = value.build()
        return value

    def values(self):
        """Yield the values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        values = []
        for value in super().values():
            values.append(self.build_lazy(value))
        return values

    def items(self):
        """Yield the keys and values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        items = []
        for key, value in super().items():
            items.append((key, self.build_lazy(value)))
        return items

    def merge(self, other: dict[str, Any]):
        """Merge another dictionary into current dictionary.

        Args:
            other (dict): Another dictionary.
        """
        default = object()

        def _merge_a_into_b(a: Any, b: Any):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    a.pop(DELETE_KEY, None)
                    return a
                if a.pop(DELETE_KEY, False):
                    b.clear()
                all_keys = list(b.keys()) + list(a.keys())
                return {
                    key: _merge_a_into_b(a.get(key, default), b.get(key, default))
                    for key in all_keys
                    if key != DELETE_KEY
                }
            else:
                return a if a is not default else b

        merged = _merge_a_into_b(copy.deepcopy(other), copy.deepcopy(self))
        self.clear()
        for key, value in merged.items():
            self[key] = value

    def __reduce_ex__(self, proto: SupportsIndex):
        # Override __reduce_ex__ to avoid `self.items` will be
        # called by CPython interpreter during pickling. See more details in
        # https://github.com/python/cpython/blob/8d61a71f9c81619e34d4a30b625922ebc83c561b/Objects/typeobject.c#L6196  # noqa: E501
        if digit_version(platform.python_version()) < digit_version("3.8"):
            return (
                self.__class__,
                ({k: v for k, v in super().items()},),
                None,
                None,
                None,
            )
        else:
            return (
                self.__class__,
                ({k: v for k, v in super().items()},),
                None,
                None,
                None,
                None,
            )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ConfigDict):
            return other.to_dict() == self.to_dict()
        elif isinstance(other, dict):
            return {k: v for k, v in self.items()} == other
        else:
            return False

    def _to_lazy_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and keep
        the ``LazyObject`` or ``LazyAttr`` object not built."""

        def _to_dict(data: dict[str, Any]):
            if isinstance(data, ConfigDict):
                return {key: _to_dict(value) for key, value in Dict.items(data)}
            elif isinstance(data, dict):
                return {key: _to_dict(value) for key, value in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(_to_dict(item) for item in data)
            else:
                return data

        return _to_dict(self)

    def to_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and
        convert the ``LazyObject`` or ``LazyAttr`` to string."""
        return _lazy2string(self, dict_type=dict)


def add_args(parser: ArgumentParser, cfg: dict[str, Any], prefix: str = "") -> ArgumentParser:
    """Add config fields into argument parser.

    Args:
        parser (ArgumentParser): Argument parser.
        cfg (dict): Config dictionary.
        prefix (str, optional): Prefix of parser argument.
            Defaults to ''.

    Returns:
        ArgumentParser: Argument parser containing config fields.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(next(iter(v))), nargs="+")
        else:
            logger.error(f"cannot parse key {prefix + k} of type {type(v)}")
    return parser


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml.
    ``Config.fromfile`` can parse a dictionary from a config file, then
    build a ``Config`` instance with the dictionary.
    The interface is the same as a dict object and also allows access config
    values as attributes.

    Args:
        cfg_dict (dict, optional): A config dictionary. Defaults to None.
        cfg_text (str, optional): Text of config. Defaults to None.
        filename (str or Path, optional): Name of config file.
            Defaults to None.
        format_python_code (bool): Whether to format Python code by yapf.
            Defaults to True.

    Here is a simple example:

    Examples:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/username/projects/engine/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/username/projects/engine/tests/data/config/a.py]
        :"
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    You can find more advance usage in the `config tutorial`_.

    .. _config tutorial: https://engine.readthedocs.io/en/latest/advanced_tutorials/config.html
    """  # noqa: E501

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        cfg_text: str | None = None,
        filename: str | Path | None = None,
        env_variables: dict[str, Any] | None = None,
        format_python_code: bool = True,
    ) -> None:
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, but got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)
        super().__setattr__("_cfg_dict", cfg_dict)
        super().__setattr__("_filename", filename)
        super().__setattr__("_format_python_code", format_python_code)
        if not hasattr(self, "_imported_names"):
            super().__setattr__("_imported_names", set())

        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("_text", text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__("_env_variables", env_variables)

    @staticmethod
    def fromfile(
        filename: str | Path,
        use_predefined_variables: bool = True,
        import_custom_modules: bool = True,
        use_environment_variables: bool = True,
        lazy_import: bool | None = None,
        format_python_code: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> Config:
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.
            kwargs (dict, optional): External variables that can be accessed
                via `_kwargs_.get('key', default)` in config files. This enables
                late-binding of variables across the config inheritance chain.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        if lazy_import is False or lazy_import is None and not Config._is_lazy_import(filename):
            cfg_dict, cfg_text, env_variables = Config._file2dict(
                filename,
                use_predefined_variables,
                use_environment_variables,
                lazy_import,
                kwargs=kwargs,
            )
            if import_custom_modules and cfg_dict.get("custom_imports", None):
                try:
                    import_modules_from_strings(**cfg_dict["custom_imports"])
                except ImportError as e:
                    err_msg = (
                        f"Failed to import custom modules from "
                        f"{cfg_dict['custom_imports']}, the current sys.path is: "
                    )
                    for p in sys.path:
                        err_msg += f"\n    {p}"
                    err_msg += (
                        "\nYou should set `PYTHONPATH` to make `sys.path` "
                        "include the directory which contains your custom "
                        "module"
                    )
                    raise ImportError(err_msg) from e
            return Config(
                cfg_dict,
                cfg_text=cfg_text,
                filename=filename,
                env_variables=env_variables,
            )
        else:
            # Enable lazy import when parsing the config.
            # Using try-except to make sure ``ConfigDict.lazy`` will be reset
            # to False. See more details about lazy in the docstring of
            # ConfigDict
            ConfigDict.lazy = True
            try:
                cfg_dict, imported_names = Config._parse_lazy_import(filename, kwargs=kwargs)
            except Exception as e:
                raise e
            finally:
                # disable lazy import to get the real type. See more details
                # about lazy in the docstring of ConfigDict
                ConfigDict.lazy = False

            cfg = Config(cfg_dict, filename=filename, format_python_code=format_python_code)
            object.__setattr__(cfg, "_imported_names", imported_names)
            return cfg

    @staticmethod
    def fromstring(cfg_str: str, file_format: str) -> Config:
        """Build a Config instance from config text.

        Args:
            cfg_str (str): Config text.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            Config: Config object generated from ``cfg_str``.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py', stacklevel=2)

        # A temporary file can not be opened a second time on Windows.
        # See https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile for more details. # noqa
        # `temp_file` is opened first in `tempfile.NamedTemporaryFile` and
        #  second in `Config.from_file`.
        # In addition, a named temporary file will be removed after closed.
        # As a workaround we set `delete=False` and close the temporary file
        # before opening again.

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)

        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)  # manually delete the temporary file
        return cfg

    @staticmethod
    def _get_base_modules(nodes: list[Any]) -> list[Any]:
        """Get base module name from parsed code.

        Args:
            nodes (list): Parsed code of the config file.

        Returns:
            list: Name of base modules.
        """

        def _get_base_module_from_with(with_nodes: list[Any]) -> list[Any]:
            """Get base module name from if statement in python file.

            Args:
                with_nodes (list): List of if statement.

            Returns:
                list: Name of base modules.
            """
            base_modules = []
            for node in with_nodes:
                assert isinstance(node, ast.ImportFrom), (
                    "Illegal syntax in config file! Only "
                    "`from ... import ...` could be implemented` in "
                    "with read_base()`"
                )
                assert node.module is not None, (
                    "Illegal syntax in config file! Syntax like "
                    "`from . import xxx` is not allowed in `with read_base()`"
                )
                base_modules.append(node.level * "." + node.module)
            return base_modules

        for idx, node in enumerate(nodes):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == BASE_KEY
            ):
                raise ConfigParsingError(
                    "The configuration file type in the inheritance chain "
                    "must match the current configuration file type, either "
                    '"lazy_import" or non-"lazy_import". You got this error '
                    f'since you use the syntax like `_base_ = "{node.targets[0].id}"` '  # noqa: E501
                    "in your config. You should use `with read_base(): ... to` "  # noqa: E501
                    "mark the inherited config file. See more information "
                    "in https://engine.readthedocs.io/en/latest/advanced_tutorials/config.html"  # noqa: E501
                )

            if not isinstance(node, ast.With):
                continue

            expr = node.items[0].context_expr
            if (
                not isinstance(expr, ast.Call)
                or not expr.func.id == "read_base"  # type: ignore
                or len(node.items) > 1
            ):
                raise ConfigParsingError(
                    "Only `read_base` context manager can be used in the config"
                )

            # The original code:
            # ```
            # with read_base():
            #     from .._base_.default_runtime import *
            # ```
            # The processed code:
            # ```
            # from .._base_.default_runtime import *
            # ```
            # As you can see, the if statement is removed and the
            # from ... import statement will be unindent
            for nested_idx, nested_node in enumerate(node.body):
                nodes.insert(idx + nested_idx + 1, nested_node)
            nodes.pop(idx)
            return _get_base_module_from_with(node.body)
        return []

    @staticmethod
    def _validate_py_syntax(filename: str) -> None:
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"There are syntax errors in config file {filename}: {e}") from e

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str) -> None:
        """Substitute predefined variables in config with actual values.

        Sometimes we want some variables in the config to be related to the
        current path or file name, etc.

        Here is an example of a typical usage scenario. When training a model,
        we define a working directory in the config that save the models and
        logs. For different configs, we expect to define different working
        directories. A common way for users is to use the config file name
        directly as part of the working directory name, e.g. for the config
        ``config_setting1.py``, the working directory is
        ``. /work_dir/config_setting1``.

        This can be easily achieved using predefined variables, which can be
        written in the config `config_setting1.py` as follows

        .. code-block:: python

           work_dir = '. /work_dir/{{ fileBasenameNoExtension }}'


        Here `{{ fileBasenameNoExtension }}` indicates the file name of the
        config (without the extension), and when the config class reads the
        config file, it will automatically parse this double-bracketed string
        to the corresponding actual value.

        .. code-block:: python

           cfg = Config.fromfile('. /config_setting1.py')
           cfg.work_dir # ". /work_dir/config_setting1"


        For details, Please refer to docs/zh_cn/advanced_tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _substitute_env_variables(filename: str, temp_config_name: str):
        """Substitute environment variables in config with actual values.

        Sometimes, we want to change some items in the config with environment
        variables. For examples, we expect to change dataset root by setting
        ``DATASET_ROOT=/dataset/root/path`` in the command line. This can be
        easily achieved by writing lines in the config as follows

        .. code-block:: python

           data_root = '{{$DATASET_ROOT:/default/dataset}}/images'


        Here, ``{{$DATASET_ROOT:/default/dataset}}`` indicates using the
        environment variable ``DATASET_ROOT`` to replace the part between
        ``{{}}``. If the ``DATASET_ROOT`` is not set, the default value
        ``/default/dataset`` will be used.

        Environment variables not only can replace items in the string, they
        can also substitute other types of data in config. In this situation,
        we can write the config as below

        .. code-block:: python

           model = dict(
               bbox_head = dict(num_classes={{'$NUM_CLASSES:80'}}))


        For details, Please refer to docs/zh_cn/tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        regexp = r"\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}"
        keys = re.findall(regexp, config_file)
        env_variables = dict()
        for var_name, value in keys:
            regexp = r"\{\{[\'\"]?\s*\$" + var_name + r"\s*\:\s*" + value + r"\s*[\'\"]?\}\}"
            if var_name in os.environ:
                value = os.environ[var_name]
                env_variables[var_name] = value
                logger.info(
                    f"Using env variable `{var_name}` with value of {value} "
                    f"to replace item in config."
                )
            if not value:
                raise KeyError(
                    f"`{var_name}` cannot be found in `os.environ`."
                    f" Please set `{var_name}` in environment or "
                    "give a default value."
                )
            config_file = re.sub(regexp, value, config_file)

        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return env_variables

    @staticmethod
    def _pre_substitute_base_vars(filename: str, temp_config_name: str) -> dict[str, Any]:
        """Preceding step for substituting variables in base config with actual
        value.

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.

        Returns:
            dict: A dictionary contains variables in base config.
        """
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _resolve_base_var(
        var_key: str, base_var_dict: dict[str, Any], base_cfg: dict[str, Any]
    ) -> Any:
        """Resolve a base variable key to its actual value from base_cfg."""
        result = base_cfg
        for key in base_var_dict[var_key].split("."):
            result = result[key]
        return result

    @staticmethod
    def _substitute_base_vars(
        cfg: Any, base_var_dict: dict[str, Any], base_cfg: dict[str, Any]
    ) -> Any:
        """Substitute base variables from strings to their actual values."""
        cfg = copy.deepcopy(cfg)
        return Config._substitute_base_vars_inplace(cfg, base_var_dict, base_cfg)

    @staticmethod
    def _substitute_base_vars_inplace(
        cfg: Any, base_var_dict: dict[str, Any], base_cfg: dict[str, Any]
    ) -> Any:
        """In-place substitution without deepcopy for recursive calls."""
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    cfg[k] = Config._resolve_base_var(v, base_var_dict, base_cfg)
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars_inplace(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars_inplace(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            for i, c in enumerate(cfg):
                cfg[i] = Config._substitute_base_vars_inplace(c, base_var_dict, base_cfg)
        elif isinstance(cfg, str) and cfg in base_var_dict:
            cfg = Config._resolve_base_var(cfg, base_var_dict, base_cfg)
        return cfg

    @staticmethod
    def _file2dict(
        filename: str,
        use_predefined_variables: bool = True,
        use_environment_variables: bool = True,
        lazy_import: bool | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            kwargs (dict, optional): External variables accessible via _kwargs_
                in config files.

        Returns:
            tuple[dict, str]: Variables dictionary and text of Config.
        """
        if lazy_import is None and Config._is_lazy_import(filename):
            raise RuntimeError(
                "The configuration file type in the inheritance chain "
                "must match the current configuration file type, either "
                '"lazy_import" or non-"lazy_import". You got this error '
                "since you use the syntax like `with read_base(): ...` "
                f"or import non-builtin module in {filename}. See more "
                "information in https://engine.readthedocs.io/en/latest/advanced_tutorials/config.html"  # noqa: E501
            )

        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        cfg_dict: dict[str, Any] = {}
        temp_config_dir: str = ""
        try:
            with tempfile.TemporaryDirectory() as temp_config_dir:
                with tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix=fileExtname, delete=False
                ) as temp_config_file:
                    temp_config_name = temp_config_file.name

                # Substitute predefined variables
                if use_predefined_variables:
                    Config._substitute_predefined_vars(filename, temp_config_name)
                else:
                    shutil.copyfile(filename, temp_config_name)
                # Substitute environment variables
                env_variables = dict()
                if use_environment_variables:
                    env_variables = Config._substitute_env_variables(
                        temp_config_name, temp_config_name
                    )
                # Substitute base variables from placeholders to strings
                base_var_dict = Config._pre_substitute_base_vars(temp_config_name, temp_config_name)

                # Handle base files
                base_cfg_dict = ConfigDict()
                cfg_text_list = list()
                for base_cfg_path in Config._get_base_files(temp_config_name):
                    base_cfg_path, scope = Config._get_cfg_path(base_cfg_path, filename)
                    _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                        filename=base_cfg_path,
                        use_predefined_variables=use_predefined_variables,
                        use_environment_variables=use_environment_variables,
                        lazy_import=lazy_import,
                        kwargs=kwargs,
                    )
                    cfg_text_list.append(_cfg_text)
                    env_variables.update(_env_variables)
                    # if len(base_cfg_dict.keys() & _cfg_dict.keys()) > 0:
                    #     logger.warning(
                    #         "Duplicate keys across bases; later ones override earlier "
                    #         f"ones: {duplicate_keys}"
                    #     )

                    # _dict_to_config_dict will do the following things:
                    # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                    # 2. Set `_scope_` for the outer dict variable for the base
                    # config.
                    # 3. Set `scope` attribute for each base variable.
                    # Different from `_scope_`, `scope` is not a key of base
                    # dict, `scope` attribute will be parsed to key `_scope_`
                    # by function `_parse_scope` only if the base variable is
                    # accessed by the current config.
                    _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                    # Merge bases in declaration order: later bases override
                    # earlier ones while keeping non-conflicting keys.
                    base_cfg_dict = Config._merge_a_into_b(_cfg_dict, base_cfg_dict)

                if filename.endswith(".py"):
                    with open(temp_config_name, encoding="utf-8") as f:
                        parsed_codes = ast.parse(f.read())
                        parsed_codes = RemoveAssignFromAST(BASE_KEY).visit(parsed_codes)
                    codeobj = compile(parsed_codes, filename, mode="exec")
                    # _kwargs_ enables late-binding of variables across inheritance chain
                    global_locals_var = {
                        BASE_KEY: base_cfg_dict,
                        "_kwargs_": ConfigDict(kwargs or {}),
                    }
                    ori_keys = set(global_locals_var.keys())
                    eval(codeobj, global_locals_var, global_locals_var)
                    cfg_dict = {
                        key: value
                        for key, value in global_locals_var.items()
                        if (key not in ori_keys and not key.startswith("__"))
                    }
                elif filename.endswith((".yml", ".yaml", ".json")):
                    cfg_dict = load(temp_config_name)
                # close temp file
                for key, value in list(cfg_dict.items()):
                    if isinstance(value, (types.FunctionType, types.ModuleType)):
                        cfg_dict.pop(key)
                temp_config_file.close()

                # If the current config accesses a base variable of base
                # configs, The ``scope`` attribute of corresponding variable
                # will be converted to the `_scope_`.
                Config._parse_scope(cfg_dict)
        except Exception as e:
            if osp.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            raise e

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f"The config file {filename} will be deprecated in the future."
            if "expected" in deprecation_info:
                warning_msg += f" Please use {deprecation_info['expected']} instead."
            if "reference" in deprecation_info:
                warning_msg += f" More information can be found at {deprecation_info['reference']}"
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

        cfg_text = filename + "\n"
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("__")}

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables

    @staticmethod
    def _parse_lazy_import(
        filename: str, kwargs: dict[str, Any] | None = None
    ) -> tuple[ConfigDict, set[Any]]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            kwargs (dict, optional): External variables accessible via _kwargs_
                in config files.

        Returns:
            tuple[dict, dict]: ``cfg_dict`` and ``imported_names``.

              - cfg_dict (dict): Variables dictionary of parsed config.
              - imported_names (set): Used to mark the names of
                imported object.
        """
        # In lazy import mode, users can use the Python syntax `import` to
        # implement inheritance between configuration files, which is easier
        # for users to understand the hierarchical relationships between
        # different configuration files.

        # Besides, users can also using `import` syntax to import corresponding
        # module which will be filled in the `type` field. It means users
        # can directly navigate to the source of the module in the
        # configuration file by clicking the `type` field.

        # To avoid really importing the third party package like `torch`
        # during import `type` object, we use `_parse_lazy_import` to parse the
        # configuration file, which will not actually trigger the import
        # process, but simply parse the imported `type`s as LazyObject objects.

        # The overall pipeline of _parse_lazy_import is:
        # 1. Parse the base module from the config file.
        #                       ||
        #                       \/
        #       base_module = ['mmdet.configs.default_runtime']
        #                       ||
        #                       \/
        # 2. recursively parse the base module and gather imported objects to
        #    a dict.
        #                       ||
        #                       \/
        #       The base_dict will be:
        #       {
        #           'mmdet.configs.default_runtime': {...}
        #           'mmdet.configs.retinanet_r50_fpn_1x_coco': {...}
        #           ...
        #       }, each item in base_dict is a dict of `LazyObject`
        # 3. parse the current config file filling the imported variable
        #    with the base_dict.
        #
        # 4. During the parsing process, all imported variable will be
        #    recorded in the `imported_names` set. These variables can be
        #    accessed, but will not be dumped by default.

        with open(filename, encoding="utf-8") as f:
            global_dict = {
                "LazyObject": LazyObject,
                "__file__": filename,
                "_kwargs_": ConfigDict(kwargs or {}),
            }
            base_dict = {}

            parsed_codes = ast.parse(f.read())
            # get the names of base modules, and remove the
            # `with read_base():'` statement
            base_modules = Config._get_base_modules(parsed_codes.body)
            base_imported_names = set()
            for base_module in base_modules:
                # If base_module means a relative import, assuming the level is
                # 2, which means the module is imported like
                # "from ..a.b import c". we must ensure that c is an
                # object `defined` in module b, and module b should not be a
                # package including `__init__` file but a single python file.
                _match = re.match(r"\.*", base_module)
                level = len(_match.group()) if _match is not None else 0
                if level > 0:
                    # Relative import
                    base_dir = osp.dirname(filename)
                    module_path = osp.join(
                        base_dir,
                        *([".."] * (level - 1)),
                        f"{base_module[level:].replace('.', '/')}.py",
                    )
                else:
                    # Absolute import
                    module_list = base_module.split(".")
                    if len(module_list) == 1:
                        raise ConfigParsingError(
                            "The imported configuration file should not be "
                            f"an independent package {module_list[0]}. Here "
                            "is an example: "
                            "`with read_base(): from mmdet.configs.retinanet_r50_fpn_1x_coco import *`"  # noqa: E501
                        )
                    else:
                        package = module_list[0]
                        root_path = get_installed_path(package)
                        module_path = f"{osp.join(root_path, *module_list[1:])}.py"  # noqa: E501
                if not osp.isfile(module_path):
                    raise ConfigParsingError(
                        f"{module_path} not found! It means that incorrect "
                        "module is defined in "
                        f"`with read_base(): = from {base_module} import ...`, please "  # noqa: E501
                        "make sure the base config module is valid "
                        "and is consistent with the prior import "
                        "logic"
                    )
                _base_cfg_dict, _base_imported_names = Config._parse_lazy_import(
                    module_path, kwargs=kwargs
                )
                base_imported_names |= _base_imported_names
                # The base_dict will be:
                # {
                #     'mmdet.configs.default_runtime': {...}
                #     'mmdet.configs.retinanet_r50_fpn_1x_coco': {...}
                #     ...
                # }
                base_dict[base_module] = _base_cfg_dict

            # `base_dict` contains all the imported modules from `base_cfg`.
            # In order to collect the specific imported module from `base_cfg`
            # before parse the current file, we using AST Transform to
            # transverse the imported module from base_cfg and merge then into
            # the global dict. After the ast transformation, most of import
            # syntax will be removed (except for the builtin import) and
            # replaced with the `LazyObject`
            transform = ImportTransformer(
                global_dict=global_dict, base_dict=base_dict, filename=filename
            )
            modified_code = transform.visit(parsed_codes)
            modified_code, abs_imported = _gather_abs_import_lazyobj(
                modified_code, filename=filename
            )
            imported_names = transform.imported_obj | abs_imported
            imported_names |= base_imported_names
            modified_code = ast.fix_missing_locations(modified_code)
            exec(compile(modified_code, filename, mode="exec"), global_dict, global_dict)

            ret: dict[str, Any] = {}
            for key, value in global_dict.items():
                if key.startswith("__") or key in ["LazyObject", "_kwargs_"]:
                    continue
                ret[key] = value
            # convert dict to ConfigDict
            cfg_dict = Config._dict_to_config_dict_lazy(ret)

            return cfg_dict, imported_names

    @staticmethod
    def _dict_to_config_dict_lazy(cfg: Any):
        """Recursively converts ``dict`` to :obj:`ConfigDict`. The only
        difference between ``_dict_to_config_dict_lazy`` and
        ``_dict_to_config_dict_lazy`` is that the former one does not consider
        the scope, and will not trigger the building of ``LazyObject``.

        Args:
            cfg (dict): Config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            cfg_dict = ConfigDict()
            for key, value in cfg.items():
                cfg_dict[key] = Config._dict_to_config_dict_lazy(value)
            return cfg_dict
        if isinstance(cfg, (tuple, list)):
            return type(cfg)(Config._dict_to_config_dict_lazy(_cfg) for _cfg in cfg)
        return cfg

    @staticmethod
    def _dict_to_config_dict(cfg: Any, scope: str | None = None, has_scope: bool = True):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.
            scope (str, optional): Scope of instance.
            has_scope (bool): Whether to add `_scope_` key to config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            if has_scope and "type" in cfg:
                has_scope = False
                if scope is not None and cfg.get("_scope_", None) is None:
                    cfg._scope_ = scope  # type: ignore
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, "scope", scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(value, scope=scope, has_scope=has_scope)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope) for _cfg in cfg
            )
        elif isinstance(cfg, list):
            cfg = [Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope) for _cfg in cfg]
        return cfg

    @staticmethod
    def _parse_scope(cfg: Any) -> None:
        """Adds ``_scope_`` to :obj:`ConfigDict` instance, which means a base
        variable.

        If the config dict already has the scope, scope will not be
        overwritten.

        Args:
            cfg (dict): Config needs to be parsed with scope.
        """
        if isinstance(cfg, ConfigDict):
            cfg._scope_ = cfg.scope
        elif isinstance(cfg, (tuple, list)):
            [Config._parse_scope(value) for value in cfg]
        else:
            return

    @staticmethod
    def _get_base_files(filename: str) -> list[Any]:
        """Get the base config file.

        Args:
            filename (str): The config file.

        Raises:
            TypeError: Name of config file.

        Returns:
            list: A list of base config.
        """
        file_format = osp.splitext(filename)[1]
        if file_format == ".py":
            Config._validate_py_syntax(filename)
            with open(filename, encoding="utf-8") as f:
                parsed_codes = ast.parse(f.read()).body

                def is_base_line(c: ast.stmt) -> bool:
                    return (
                        isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY
                    )

                base_code = next((c for c in parsed_codes if is_base_line(c)), None)
                if base_code is not None:
                    assert isinstance(base_code, ast.Assign)
                    base_expr = ast.Expression(body=base_code.value)
                    base_files = eval(compile(base_expr, "", mode="eval"))
                else:
                    base_files = []
        elif file_format in (".yml", ".yaml", ".json"):
            cfg_dict = load(filename)
            base_files = cfg_dict.pop(BASE_KEY, [])
        else:
            raise ConfigParsingError(
                f"The config type should be py, json, yaml or yml, but got {file_format}"
            )
        base_files = base_files if isinstance(base_files, list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str, filename: str) -> tuple[str, str | None]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        if "::" in cfg_path:
            # `cfg_path` startswith '::' means an external config path.
            # Get package name and relative config path.
            scope = cfg_path.partition("::")[0]
            package, cfg_path = _get_package_and_cfg_path(cfg_path)

            if not is_installed(package):
                raise ModuleNotFoundError(
                    f"{package} is not installed, please install {package} manually"
                )

            # Get installed package path.
            package_path = get_installed_path(package)
            try:
                # Get config path from meta file.
                cfg_path = _get_external_cfg_path(package_path, cfg_path)
            except ValueError:
                # Since base config does not have a metafile, it should be
                # concatenated with package path and relative config path.
                cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
            except FileNotFoundError as e:
                raise e
            return cfg_path, scope
        else:
            # Get local config path.
            cfg_dir = osp.dirname(filename)
            cfg_path = osp.join(cfg_dir, cfg_path)
            return cfg_path, None

    @staticmethod
    def _merge_a_into_b(a: dict[str, Any], b: Any, allow_list_keys: bool = False) -> Any:
        """Merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k_int = int(k)
                if len(b) <= k_int:
                    raise KeyError(f"Index {k_int} exceeds the length of list {b}")
                b[k_int] = Config._merge_a_into_b(v, b[k_int], allow_list_keys)
            elif isinstance(v, dict):
                assert not isinstance(b, list)
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: tuple[type, ...] | type = (
                        (dict, list) if allow_list_keys else dict
                    )
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                assert not isinstance(b, list)
                b[k] = v
        return b

    @staticmethod
    def auto_argparser(description: str | None = None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg.to_dict())
        return parser, cfg

    @property
    def filename(self) -> str:
        """Get file name of config."""
        return self._filename

    @property
    def text(self) -> str:
        """Get config text."""
        return self._text

    @property
    def env_variables(self) -> dict[str, Any]:
        """Get used environment variables."""
        return self._env_variables

    @property
    def pretty_text(self) -> str:
        """Get formatted python config text."""
        import yapf
        from yapf.yapflib.yapf_api import FormatCode

        indent = 4

        def _indent(s_: str, num_spaces: int):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k: Any, v: Any, use_mapping: bool = False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list_tuple(k: Any, v: Any, use_mapping: bool = False):
            if isinstance(v, list):
                left = "["
                right = "]"
            else:
                left = "("
                right = ")"

            v_str = f"{left}\n"
            # check if all items in the list are dict
            for item in v:
                if isinstance(item, dict):
                    v_str += f"dict({_indent(_format_dict(item), indent)}),\n"
                elif isinstance(item, tuple):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: E501
                elif isinstance(item, list):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: E501
                elif isinstance(item, str):
                    v_str += f"{_indent(repr(item), indent)},\n"
                else:
                    v_str += str(item) + ",\n"
            if k is None:
                return _indent(v_str, indent) + right
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent) + right
            return attr_str

        def _contain_invalid_identifier(dict_str: Any):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict: dict[str, Any], outest_level: bool = False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(sorted(input_dict.items(), key=lambda x: str(x[0]))):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _format_list_tuple(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        if self._format_python_code:
            # copied from setup.cfg
            yapf_style = dict(
                based_on_style="pep8",
                blank_line_before_nested_class_or_def=True,
                split_before_expression_after_opening_paren=True,
            )
            try:
                if digit_version(yapf.__version__) >= digit_version("0.40.2"):
                    text, _ = FormatCode(text, style_config=yapf_style)
                else:
                    text, _ = FormatCode(text, style_config=yapf_style, verify=True)
            except Exception as e:  # noqa: BLE001
                raise SyntaxError(
                    f"Failed to format the config file, please check the syntax of: \n{text}"
                ) from e
        return text

    def __repr__(self) -> str:
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self) -> int:
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cfg_dict)

    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], str | None, str | None, dict[str, Any], bool, set[Any]]:
        state = (
            self._cfg_dict,
            self._filename,
            self._text,
            self._env_variables,
            self._format_python_code,
            self._imported_names,
        )
        return state

    def __deepcopy__(self, memo: dict[int, object]):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        super(Config, other).__setattr__("_cfg_dict", self._cfg_dict.copy())

        return other

    copy = __copy__

    def __setstate__(
        self, state: tuple[dict[str, Any], str | None, str | None, dict[str, Any], bool, set[Any]]
    ) -> None:
        super().__setattr__("_cfg_dict", state[0])
        super().__setattr__("_filename", state[1])
        super().__setattr__("_text", state[2])
        super().__setattr__("_env_variables", state[3])
        super().__setattr__("_format_python_code", state[4])
        super().__setattr__("_imported_names", state[5])

    def dump(self, file: str | Path | None = None, keys: list[str] | None = None):
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.
            keys (list, optional): Only dump these keys. Defaults to None (all keys).

        Returns:
            str or None: Config text.
        """
        import json

        file = str(file) if isinstance(file, Path) else file
        cfg_dict = self.to_dict()

        if keys:
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in keys}

        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
            else:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith(".py"):
            # save as json to avoid yapf slow formatting
            json_file = file.replace(".py", ".json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, indent=2, default=str)
        else:
            file_format = file.split(".")[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options: dict[str, Any], allow_list_keys: bool = True) -> None:
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
                are allowed in ``options`` and will replace the element of the
                corresponding index in the config if the config is a list.
                Defaults to True.

        Examples:
            >>> from engine import Config
            >>> #  Merge dictionary element
            >>> options = {'model.backbone.depth': 50, 'model.backbone.with_cp': True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg._cfg_dict
            {'model': {'backbone': {'type': 'ResNet', 'depth': 50, 'with_cp': True}}}
            >>> # Merge list element
            >>> cfg = Config(
            >>>     dict(pipeline=[dict(type='LoadImage'),
            >>>                    dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg._cfg_dict
            {'pipeline': [{'type': 'SelfLoadImage'}, {'type': 'LoadAnnotations'}]}
        """  # noqa: E501
        option_cfg_dict: dict[str, Any] = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__("_cfg_dict")
        super().__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys),
        )

    @staticmethod
    def diff(cfg1: str  | Config, cfg2: str  | Config) -> str:
        if isinstance(cfg1, str):
            cfg1 = Config.fromfile(cfg1)

        if isinstance(cfg2, str):
            cfg2 = Config.fromfile(cfg2)

        res = difflib.unified_diff(cfg1.pretty_text.split("\n"), cfg2.pretty_text.split("\n"))

        # Convert into rich format for better visualization
        console = Console()
        text = Text()
        for line in res:
            if line.startswith("+"):
                color = "bright_green"
            elif line.startswith("-"):
                color = "bright_red"
            else:
                color = "bright_white"
            _text = Text(line + "\n")
            _text.stylize(color)
            text.append(_text)

        with console.capture() as capture:
            console.print(text)

        return capture.get()

    @staticmethod
    def _is_lazy_import(filename: str) -> bool:
        if not filename.endswith(".py"):
            return False
        with open(filename, encoding="utf-8") as f:
            codes_str = f.read()
            parsed_codes = ast.parse(codes_str)
        for node in ast.walk(parsed_codes):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == BASE_KEY
            ):
                return False

            if isinstance(node, ast.With):
                expr = node.items[0].context_expr
                if not isinstance(expr, ast.Call) or not expr.func.id == "read_base":  # noqa: E501  # type: ignore
                    raise ConfigParsingError(
                        "Only `read_base` context manager can be used in the config"
                    )
                return True
            if isinstance(node, ast.ImportFrom):
                # relative import -> lazy_import
                if node.level != 0:
                    return True
                # Skip checking when using `engine.config` in cfg file
                if (
                    node.module == "engine"
                    and len(node.names) == 1
                    and node.names[0].name == "Config"
                ):
                    continue
                if not isinstance(node.module, str):
                    continue
                # non-builtin module -> lazy_import
                if not _is_builtin_module(node.module):
                    return True
            if isinstance(node, ast.Import):
                for alias_node in node.names:
                    if not _is_builtin_module(alias_node.name):
                        return True
        return False

    def _to_lazy_dict(self, keep_imported: bool = False) -> dict[str, Any]:
        """Convert config object to dictionary with lazy object, and filter the
        imported object."""
        res = self._cfg_dict._to_lazy_dict()
        if hasattr(self, "_imported_names") and not keep_imported:
            res = {key: value for key, value in res.items() if key not in self._imported_names}
        return res

    def to_dict(self, keep_imported: bool = False):
        """Convert all data in the config to a builtin ``dict``.

        Args:
            keep_imported (bool): Whether to keep the imported field.
                Defaults to False

        If you import third-party objects in the config file, all imported
        objects will be converted to a string like ``torch.optim.SGD``
        """
        cfg_dict = self._cfg_dict.to_dict()
        if hasattr(self, "_imported_names") and not keep_imported:
            cfg_dict = {
                key: value for key, value in cfg_dict.items() if key not in self._imported_names
            }
        return cfg_dict


class DictAction(Action):
    """Argparse action to split an argument into KEY=VALUE form on the first =
    and append to a dictionary.

    List options can be passed as comma separated values, i.e 'KEY=V1,V2,V3',
    or with explicit brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested
    brackets to build list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> int | float | bool | Any:
        """Parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> list[Any] | tuple[Any, ...] | Any:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string: str):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                    (char == ",")
                    and (pre.count("(") == pre.count(")"))
                    and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (str | Sequence[Any] | None): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split("=", maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


@contextmanager
def read_base() -> Generator[None, None, None]:
    """Context manager to mark the base config.

    The pure Python-style configuration file allows you to use the import
    syntax. However, it is important to note that you need to import the base
    configuration file within the context of ``read_base``, and import other
    dependencies outside of it.

    You can see more usage of Python-style configuration in the `tutorial`_

    .. _tutorial: https://engine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta
    """  # noqa: E501
    yield
