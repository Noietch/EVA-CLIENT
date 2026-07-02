"""Vendored mmengine-style Registry + build_from_cfg + ManagerMixin + DefaultScope.

Merged from upstream engine/registry/{registry,build_functions,default_scope}.py
and engine/utils/manager.py. Pruned of model build (checkpoint loading) and
training-only knobs.
"""

from __future__ import annotations

import copy
import inspect
import logging
import sys
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, TypeVar

from rich.console import Console
from rich.table import Table

from ._utils import get_object_from_string, is_seq_of
from .config import Config, ConfigDict

MODULE2PACKAGE: dict[str, str] = {}

# ===== from utils/manager.py =====

_lock = threading.RLock()
T = TypeVar("T")


def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()


class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain the ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args: Any) -> None:
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert "name" in params_names, f"{cls} must have the `name` argument"
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = "", **kwargs: Any) -> None:
        assert isinstance(name, str) and name, "name argument must be an non-empty string."
        self._instance_name = name

    @classmethod
    def get_instance(cls: type[T], name: str, **kwargs: Any) -> T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), f"type of name should be str, but got {type(cls)}"
        instance_dict = cls._instance_dict  # type: ignore
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)  # type: ignore
            instance_dict[name] = instance  # type: ignore
        elif kwargs:
            warnings.warn(
                f"{cls} instance named of {name} has been created, "
                "the method `get_instance` should not accept any other "
                "arguments", stacklevel=2
            )
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f"Before calling {cls.__name__}.get_current_instance(), you "
                "should call get_instance(name=xxx) at least once."
            )
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name

# ===== from default_scope.py =====

class DefaultScope(ManagerMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Consider the case of resetting the current ``Registry`` by
    ``default_scope`` in the internal module which cannot access runner
    directly, it is difficult to get the ``default_scope`` defined in
    ``Runner``. However, if ``Runner`` created ``DefaultScope`` instance
    by given ``default_scope``, the internal module can get
    ``default_scope`` by ``DefaultScope.get_current_instance`` everywhere.

    Args:
        name (str): Name of default scope for global access.
        scope_name (str): Scope of current task.

    Examples:
        >>> from eva.engine.model import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('task', scope_name='mmdet')
        >>> # Get default scope globally.
        >>> scope_name = DefaultScope.get_instance('task').scope_name
    """

    def __init__(self, name: str, scope_name: str) -> None:
        super().__init__(name)
        assert isinstance(scope_name, str), f"scope_name should be a string, but got {scope_name}"
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> DefaultScope | None:
        """Get latest created default scope.

        Since default_scope is an optional argument for ``Registry.build``.
        ``get_current_instance`` should return ``None`` if there is no
        ``DefaultScope`` created.

        Examples:
            >>> default_scope = DefaultScope.get_current_instance()
            >>> # There is no `DefaultScope` created yet,
            >>> # `get_current_instance` return `None`.
            >>> default_scope = DefaultScope.get_instance(
            >>>     'instance_name', scope_name='engine')
            >>> default_scope.scope_name
            engine
            >>> default_scope = DefaultScope.get_current_instance()
            >>> default_scope.scope_name
            engine

        Returns:
            DefaultScope | None: Return None If there has not been
            ``DefaultScope`` instance created yet, otherwise return the
            latest created DefaultScope instance.
        """
        _accquire_lock()
        if cls._instance_dict:
            instance = super().get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: str | None) -> Generator:
        """Overwrite the current default scope with `scope_name`"""
        if scope_name is None:
            yield
        else:
            tmp = copy.deepcopy(cls._instance_dict)
            # To avoid create an instance with the same name.
            time.sleep(1e-6)
            cls.get_instance(f"overwrite-{time.time()}", scope_name=scope_name)
            try:
                yield
            finally:
                cls._instance_dict = tmp

# ===== Registry =====

class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Args:
        name (str): Registry name.
        build_func (callable, optional): A function to construct instance
            from Registry. :func:`build_from_cfg` is used if neither ``parent``
            or ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Defaults to None.
        parent (:obj:`Registry`, optional): Parent registry. The class
            registered in children registry could be built from parent.
            Defaults to None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Defaults to None.
        locations (list): The locations to import the modules registered
            in this registry. Defaults to [].
            New in version 0.4.0.

    Examples:
        >>> # define a registry
        >>> MODELS = Registry('models')
        >>> # registry the `ResNet` to `MODELS`
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> # build model from `MODELS`
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

        >>> # hierarchical registry
        >>> DETECTORS = Registry('detectors', parent=MODELS, scope='det')
        >>> @DETECTORS.register_module()
        >>> class FasterRCNN:
        >>>     pass
        >>> fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

        >>> # add locations to enable auto import
        >>> DETECTORS = Registry('detectors', parent=MODELS,
        >>>     scope='det', locations=['det.models.detectors'])
        >>> # define this class in 'det.models.detectors'
        >>> @DETECTORS.register_module()
        >>> class MaskRCNN:
        >>>     pass
        >>> # The registry will auto import det.models.detectors.MaskRCNN
        >>> fasterrcnn = DETECTORS.build(dict(type='det.MaskRCNN'))

    More advanced usages can be found at
    https://engine.readthedocs.io/en/latest/advanced_tutorials/registry.html.
    """

    def __init__(
        self,
        name: str,
        build_func: Callable[..., Any] | None = None,
        parent: Registry | None = None,
        scope: str | None = None,
        locations: list[str] | None = None,
    ) -> None:

        self._name = name
        self._module_dict: dict[str, type[Any]] = dict()
        self._children: dict[str, Registry] = dict()
        self._locations = locations if locations is not None else []
        self._imported = False

        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope()

        # See https://mypy.readthedocs.io/en/stable/common_issues.html#
        # variables-vs-type-aliases for the use
        self.parent: Registry | None
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @staticmethod
    def infer_scope() -> str:
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Returns:
            str: The inferred scope name.

        Examples:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> # The scope of ``ResNet`` will be ``mmdet``.
        """

        # `sys._getframe` returns the frame object that many calls below the
        # top of the stack. The call stack for `infer_scope` can be listed as
        # follow:
        # frame-0: `infer_scope` itself
        # frame-1: `__init__` of `Registry` which calls the `infer_scope`
        # frame-2: Where the `Registry(...)` is called
        module = inspect.getmodule(sys._getframe(2))
        if module is not None:
            filename = module.__name__
            split_filename = filename.split(".")
            scope = split_filename[0]
        else:
            # use "engine" to handle some cases which can not infer the scope
            # like initializing Registry in interactive mode
            scope = "engine"
            logger.warning(
                'set scope as "engine" when scope can not be inferred. You '
                'can silence this warning by passing a "scope" argument to '
                'Registry like `Registry(name, scope="toy")`'
            )

        return scope

    @staticmethod
    def split_scope_key(key: str) -> tuple[str | None, str]:
        """Split scope and key.

        The first scope will be split from key.

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    @contextmanager
    def switch_scope_and_registry(self, scope: str | None) -> Generator:
        """Temporarily switch default scope to the target scope, and get the
        corresponding registry.

        If the registry of the corresponding scope exists, yield the
        registry, otherwise yield the current itself.

        Args:
            scope (str, optional): The target scope.

        Examples:
            >>> from eva.engine.registry import Registry, DefaultScope, MODELS
            >>> import time
            >>> # External Registry
            >>> MMDET_MODELS = Registry('mmdet_model', scope='mmdet',
            >>>     parent=MODELS)
            >>> MMCLS_MODELS = Registry('mmcls_model', scope='mmcls',
            >>>     parent=MODELS)
            >>> # Local Registry
            >>> CUSTOM_MODELS = Registry('custom_model', scope='custom',
            >>>     parent=MODELS)
            >>>
            >>> # Initiate DefaultScope
            >>> DefaultScope.get_instance(f'scope_{time.time()}',
            >>>     scope_name='custom')
            >>> # Check default scope
            >>> DefaultScope.get_current_instance().scope_name
            custom
            >>> # Switch to mmcls scope and get `MMCLS_MODELS` registry.
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmcls
            >>>     registry.scope
            mmcls
            >>> # Nested switch scope
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmdet') as mmdet_registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmdet
            >>>     mmdet_registry.scope
            mmdet
            >>>     with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as mmcls_registry:
            >>>         DefaultScope.get_current_instance().scope_name
            mmcls
            >>>         mmcls_registry.scope
            mmcls
            >>>
            >>> # Check switch back to original scope.
            >>> DefaultScope.get_current_instance().scope_name
            custom
        """  # noqa: E501

        # Switch to the given scope temporarily. If the corresponding registry
        # can be found in root registry, return the registry under the scope,
        # otherwise return the registry itself.
        with DefaultScope.overwrite_default_scope(scope):
            # Get the global default scope
            default_scope = DefaultScope.get_current_instance()
            # Get registry by scope
            if default_scope is not None:
                scope_name = default_scope.scope_name
                try:
                    import_module(f"{scope_name}.registry")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if scope in MODULE2PACKAGE:
                        logger.warning(
                            f"{scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{scope}, Please install {scope} by "
                            f"`pip install {MODULE2PACKAGE[scope]}.",
                        )
                    else:
                        logger.warning(
                            f"Failed to import `{scope}.registry` "
                            f"make sure the registry.py exists in `{scope}` "
                            "package.",
                        )
                root = self._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    # if `default_scope` can not be found, fallback to argument
                    # `registry`
                    logger.warning(
                        f'Failed to search registry with scope "{scope_name}" '
                        f'in the "{root.name}" registry tree. '
                        f'As a workaround, the current "{self.name}" registry '
                        f'in "{self.scope}" is used to build instance. This '
                        "may cause unexpected failure when running the built "
                        f'modules. Please check whether "{scope_name}" is a '
                        "correct scope, or whether the registry is "
                        "initialized.",
                    )
                    registry = self
            # If there is no built default scope, just return current registry.
            else:
                registry = self
            yield registry

    def _get_root_registry(self) -> Registry:
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def import_from_location(self) -> None:
        """Import modules from the pre-defined locations in self._location."""
        if not self._imported:
            # Avoid circular import

            # avoid BC breaking
            if len(self._locations) == 0 and self.scope in MODULE2PACKAGE:
                logger.debug(
                    f'The "{self.name}" registry in {self.scope} did not '
                    "set import location. Fallback to call "
                    f"`{self.scope}.utils.register_all_modules` "
                    "instead.",
                )
                try:
                    module = import_module(f"{self.scope}.utils")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if self.scope in MODULE2PACKAGE:
                        logger.warning(
                            f"{self.scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{self.scope}, Please install {self.scope} by "
                            f"`pip install {MODULE2PACKAGE[self.scope]}."
                        )
                    else:
                        logger.warning(
                            f"Failed to import {self.scope} and register "
                            "its modules, please make sure you "
                            "have registered the module manually."
                        )
                else:
                    # The import errors triggered during the registration
                    # may be more complex, here just throwing
                    # the error to avoid causing more implicit registry errors
                    # like `xxx`` not found in `yyy` registry.
                    module.register_all_modules(False)  # type: ignore

            for loc in self._locations:
                import_module(loc)
                logger.debug(
                    f"Modules of {self.scope}'s {self.name} registry have been "
                    f"automatically imported from {loc}",
                )
            self._imported = True

    def get(self, key: str) -> type | None:
        """Get the registry record.

        If `key`` represents the whole object name with its module
        information, for example, `engine.model.BaseModel`, ``get``
        will directly return the class object :class:`BaseModel`.

        Otherwise, it will first parse ``key`` and check whether it
        contains a scope name. The logic to search for ``key``:

        - ``key`` does not contain a scope name, i.e., it is purely a module
          name like "ResNet": :meth:`get` will search for ``ResNet`` from the
          current registry to its parent or ancestors until finding it.

        - ``key`` contains a scope name and it is equal to the scope of the
          current registry (e.g., "mmcls"), e.g., "mmcls.ResNet": :meth:`get`
          will only search for ``ResNet`` in the current registry.

        - ``key`` contains a scope name and it is not equal to the scope of
          the current registry (e.g., "mmdet"), e.g., "mmcls.FCNet": If the
          scope exists in its children, :meth:`get` will get "FCNet" from
          them. If not, :meth:`get` will first get the root registry and root
          registry call its own :meth:`get` method.

        Args:
            key (str): Name of the registered item, e.g., the class name in
                string format.

        Returns:
            Type or None: Return the corresponding class if ``key`` exists,
            otherwise return None.

        Examples:
            >>> # define a registry
            >>> MODELS = Registry('models')
            >>> # register `ResNet` to `MODELS`
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet_cls = MODELS.get('ResNet')

            >>> # hierarchical registry
            >>> DETECTORS = Registry('detector', parent=MODELS, scope='det')
            >>> # `ResNet` does not exist in `DETECTORS` but `get` method
            >>> # will try to search from its parents or ancestors
            >>> resnet_cls = DETECTORS.get('ResNet')
            >>> CLASSIFIER = Registry('classifier', parent=MODELS, scope='cls')
            >>> @CLASSIFIER.register_module()
            >>> class MobileNet:
            >>>     pass
            >>> # `get` from its sibling registries
            >>> mobilenet_cls = DETECTORS.get('cls.MobileNet')
        """
        # Avoid circular import

        if not isinstance(key, str):
            raise TypeError(f"The key argument of `Registry.get` must be a str, got {type(key)}")

        scope, real_key = self.split_scope_key(key)
        obj_cls = None
        registry_name = self.name
        scope_name = self.scope

        # lazy import the modules to register them into the registry
        self.import_from_location()

        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                obj_cls = self._module_dict[real_key]
            elif scope is None:
                # try to get the target from its parent or ancestors
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        obj_cls = parent._module_dict[real_key]
                        registry_name = parent.name
                        scope_name = parent.scope
                        break
                    parent = parent.parent
        else:
            # import the registry to add the nodes into the registry tree
            try:
                import_module(f"{scope}.registry")
                logger.debug(
                    f"Registry node of {scope} has been automatically imported.",
                )
            except (ImportError, AttributeError, ModuleNotFoundError):
                logger.warning(
                    f"Cannot auto import {scope}.registry, please check "
                    f'whether the package "{scope}" is installed correctly '
                    "or import the registry manually.",
                )
            # get from self._children
            if scope in self._children:
                obj_cls = self._children[scope].get(real_key)
                registry_name = self._children[scope].name
                scope_name = scope
            else:
                root = self._get_root_registry()

                if scope != root._scope and scope not in root._children:
                    # If not skip directly, `root.get(key)` will recursively
                    # call itself until RecursionError is thrown.
                    pass
                else:
                    obj_cls = root.get(key)

        if obj_cls is None:
            # Actually, it's strange to implement this `try ... except` to
            # get the object by its name in `Registry.get`. However, If we
            # want to build the model using a configuration like
            # `dict(type='engine.model.BaseModel')`, which can
            # be dumped by lazy import config, we need this code snippet
            # for `Registry.get` to work.
            try:
                obj_cls = get_object_from_string(key)
            except Exception as e:
                raise RuntimeError(f"Failed to get {key}") from e

        if obj_cls is not None:
            # For some rare cases (e.g. obj_cls is a partial function), obj_cls
            # doesn't have `__name__`. Use default value to prevent error
            cls_name = getattr(obj_cls, "__name__", str(obj_cls))
            logger.debug(
                f'Get class `{cls_name}` from "{registry_name}" registry in "{scope_name}"',
            )

        return obj_cls

    def _search_child(self, scope: str) -> Registry | None:
        """Depth-first search for the corresponding registry in its children.

        Note that the method only search for the corresponding registry from
        the current registry. Therefore, if we want to search from the root
        registry, :meth:`_get_root_registry` should be called to get the
        root registry first.

        Args:
            scope (str): The scope name used for searching for its
                corresponding registry.

        Returns:
            Registry or None: Return the corresponding registry if ``scope``
            exists, otherwise return None.
        """
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._search_child(scope)
            if registry is not None:
                return registry

        return None

    def build_from_cfg(self, cfg: dict, *args: Any, **kwargs: Any) -> Any:
        """Build an instance from a {type: ...} dict via the configured build_func.

        Engine-native call style; equivalent to the module-level ``build_from_cfg``
        but routed through this registry instance.
        """
        return self.build_func(cfg, *args, **kwargs, registry=self)

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Positional-args build path used by CLIENT callsites.

        Looks up the class/function registered under ``key`` and calls it with
        the supplied ``*args``/``**kwargs``. This is the dominant style across
        the project's ROBOT/POLICY/STRATEGY/KINEMATICS/TRANSPORT registries.
        For the engine-native ``dict(type=..., ...)`` form, use
        :meth:`build_from_cfg`.
        """
        if key not in self._module_dict:
            available = ", ".join(sorted(self._module_dict)) or "(none)"
            raise KeyError(f"Unknown {self.name} '{key}'; available: {available}")
        return self._module_dict[key](*args, **kwargs)

    def _add_child(self, registry: Registry) -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(
        self,
        module: type,
        module_name: str | list[str] | None = None,
        force: bool = False,
    ) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f"module must be Callable, but got {type(module)}")

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(
                    f"{name} is already registered in {self.name} "
                    f"at {existed_module.__module__}"
                )
            self._module_dict[name] = module

    # ----- legacy/compatibility helpers used by CLIENT callsites -----

    def register(self, key: str) -> Callable[[type | Callable], type | Callable]:
        """Decorator-style register under ``key``.

        Equivalent to ``register_module(name=key, force=False)``. Kept for
        backward compat with the lightweight Registry api previously used by
        ROBOT_REGISTRY/POLICY_REGISTRY/STRATEGY_REGISTRY/etc.
        """

        def decorator(obj: type | Callable) -> type | Callable:
            self._register_module(obj, key, force=False)  # pyright: ignore[reportArgumentType]
            return obj

        return decorator


    def available(self) -> list[str]:
        """Return the sorted list of registered keys (legacy api)."""
        return sorted(self._module_dict)

    def register_module(
        self,
        name: str | list[str] | None = None,
        force: bool = False,
        module: type | None = None,
    ) -> type | Callable:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                f"name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module: type[Any]):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

# ===== from build_functions.py =====

logger = logging.getLogger(__name__)


def build_from_cfg(
    cfg: dict | ConfigDict | Config,
    registry: Registry,
    default_args: dict | ConfigDict | Config | None = None,
) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    At least one of ``cfg`` and ``default_args`` must contain the key ``type``,
    which should be either a string registered in ``registry`` or a callable.
    When both contain ``type``, the value in ``cfg`` wins. The remaining keys
    are passed to the constructor / callable as keyword arguments.

    Args:
        cfg: Config dict with at least a ``type`` key.
        registry: Registry to look up the class/function from.
        default_args: Default kwargs merged under cfg.

    Returns:
        The constructed object.
    """

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}")

    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                f'`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )

    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a Registry object, but got {type(registry)}")

    if not (isinstance(default_args, (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            f"default_args should be a dict, ConfigDict, Config or None, "
            f"but got {type(default_args)}"
        )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Optional _scope_ honored for engine compatibility; CLIENT uses a single scope.
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.scope}::{registry.name} registry. "
                    f"Available: {', '.join(sorted(registry._module_dict)) or '(none)'}"
                )
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        if inspect.isclass(obj_cls) and issubclass(obj_cls, ManagerMixin):
            obj = obj_cls.get_instance(**args)  # type: ignore[attr-defined]
        else:
            obj = obj_cls(**args)  # pyright: ignore[reportCallIssue]

        if inspect.isclass(obj_cls) or inspect.isfunction(obj_cls) or inspect.ismethod(obj_cls):
            logger.debug(
                "An `%s` instance is built from registry (impl: %s)",
                obj_cls.__name__,
                obj_cls.__module__,
            )
        else:
            logger.debug("An instance is built from registry; constructor: %s", obj_cls)
        return obj
