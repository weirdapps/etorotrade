"""Coverage tests for yahoofinance.utils.dependency_injection module."""

import pytest

from yahoofinance.core.errors import ValidationError
from yahoofinance.utils.dependency_injection import (
    Registry,
    inject,
    lazy_import,
    provides,
)


class TestRegistry:
    """Tests for Registry class."""

    def test_init(self):
        reg = Registry()
        assert reg._registry == {}
        assert reg._instances == {}

    def test_register_directly(self):
        reg = Registry()
        factory = lambda: "result"
        returned = reg.register("key1", factory)
        assert returned is factory
        assert "key1" in reg._registry

    def test_register_as_decorator(self):
        reg = Registry()

        @reg.register("key2")
        def factory():
            return "decorated_result"

        assert "key2" in reg._registry
        assert reg.resolve("key2") == "decorated_result"

    def test_register_decorator_without_factory(self):
        reg = Registry()

        @reg.register("key3")
        def my_factory():
            return 42

        assert reg.resolve("key3") == 42

    def test_register_instance(self):
        reg = Registry()
        obj = {"data": 123}
        returned = reg.register_instance("singleton", obj)
        assert returned is obj
        assert reg.resolve("singleton") is obj

    def test_resolve_singleton_takes_priority(self):
        reg = Registry()
        reg.register("key", lambda: "from_factory")
        reg.register_instance("key", "from_instance")
        assert reg.resolve("key") == "from_instance"

    def test_resolve_factory_with_kwargs(self):
        reg = Registry()
        reg.register("key", lambda x=1, y=2: x + y)
        assert reg.resolve("key", x=10, y=20) == 30

    def test_resolve_factory_error(self):
        reg = Registry()
        reg.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        def bad_factory():
            raise RuntimeError("boom")

        reg.register("bad", bad_factory)
        with pytest.raises(ValidationError, match="Failed to resolve"):
            reg.resolve("bad")

    def test_resolve_missing_key(self):
        reg = Registry()
        with pytest.raises(ValidationError, match="No component registered"):
            reg.resolve("nonexistent")

    def test_resolve_all(self):
        reg = Registry()
        reg.register("a", lambda: "factory_a")
        reg.register_instance("b", "instance_b")
        result = reg.resolve_all()
        assert result["a"] == "factory_a"
        assert result["b"] == "instance_b"

    def test_resolve_all_singletons_not_overwritten(self):
        reg = Registry()
        reg.register("key", lambda: "factory")
        reg.register_instance("key", "instance")
        result = reg.resolve_all()
        assert result["key"] == "instance"

    def test_resolve_all_errors(self):
        reg = Registry()

        def bad():
            raise RuntimeError("fail")

        reg.register("bad", bad)
        reg.register("good", lambda: "ok")
        result = reg.resolve_all()
        assert "good" in result
        assert "bad" not in result

    def test_set_instance(self):
        reg = Registry()
        reg.set_instance("key", "value")
        assert reg.resolve("key") == "value"

    def test_clear_instances(self):
        reg = Registry()
        reg.register_instance("a", 1)
        reg.register_instance("b", 2)
        reg.clear_instances()
        assert reg._instances == {}

    def test_reset(self):
        reg = Registry()
        reg.register("a", lambda: 1)
        reg.register_instance("b", 2)
        reg.reset()
        assert reg._registry == {}
        assert reg._instances == {}

    def test_get_keys(self):
        reg = Registry()
        reg.register("a", lambda: 1)
        reg.register_instance("b", 2)
        keys = reg.get_keys()
        assert "a" in keys
        assert "b" in keys

    def test_get_keys_deduplication(self):
        reg = Registry()
        reg.register("key", lambda: 1)
        reg.register_instance("key", 2)
        keys = reg.get_keys()
        assert keys.count("key") == 1

    def test_has_key(self):
        reg = Registry()
        assert reg.has_key("x") is False
        reg.register("x", lambda: 1)
        assert reg.has_key("x") is True

    def test_has_key_instance(self):
        reg = Registry()
        reg.register_instance("y", "val")
        assert reg.has_key("y") is True


class TestInject:
    """Tests for inject decorator."""

    def test_inject_resolves_component(self):
        reg = Registry()
        reg.register("my_service", lambda: "service_value")

        # Temporarily use the global registry
        import yahoofinance.utils.dependency_injection as di

        original_registry = di.registry
        di.registry = reg

        try:

            @inject("my_service")
            def use_service(my_service=None):
                return my_service

            result = use_service()
            assert result == "service_value"
        finally:
            di.registry = original_registry

    def test_inject_does_not_overwrite_explicit_param(self):
        reg = Registry()
        reg.register("my_service", lambda: "auto_value")

        import yahoofinance.utils.dependency_injection as di

        original_registry = di.registry
        di.registry = reg

        try:

            @inject("my_service")
            def use_service(my_service=None):
                return my_service

            result = use_service(my_service="explicit_value")
            assert result == "explicit_value"
        finally:
            di.registry = original_registry

    def test_inject_handles_resolution_failure(self):
        reg = Registry()
        # don't register anything

        import yahoofinance.utils.dependency_injection as di

        original_registry = di.registry
        di.registry = reg

        try:

            @inject("missing_component")
            def use_service(missing_component=None):
                return missing_component

            # Should not raise, just leave param as None
            result = use_service()
            assert result is None
        finally:
            di.registry = original_registry

    def test_inject_with_callable_key(self):
        """Test inject when called with a callable instead of string key."""
        import yahoofinance.utils.dependency_injection as di

        original_registry = di.registry
        di.registry = Registry()

        try:
            # This tests the direct decorator path (callable component_key)
            @inject
            def my_func(my_func=None):
                return my_func

            # No component registered, should not fail
            result = my_func()
            assert result is None
        finally:
            di.registry = original_registry


class TestProvides:
    """Tests for provides decorator."""

    def test_provides_registers_instance(self):
        import yahoofinance.utils.dependency_injection as di

        original_registry = di.registry
        di.registry = Registry()

        try:

            @provides("my_config")
            def create_config():
                return {"key": "value"}

            result = create_config()
            assert result == {"key": "value"}
            assert di.registry.resolve("my_config") == {"key": "value"}
        finally:
            di.registry = original_registry


class TestLazyImport:
    """Tests for lazy_import function."""

    def test_import_module(self):
        result = lazy_import("json")
        import json

        assert result is json

    def test_import_class_from_module(self):
        result = lazy_import("json", "JSONDecodeError")
        import json

        assert result is json.JSONDecodeError

    def test_import_nonexistent_module(self):
        result = lazy_import("nonexistent_module_xyz")
        # Should return a callable that raises ImportError
        assert callable(result)
        with pytest.raises(ImportError):
            result()

    def test_import_nonexistent_class(self):
        result = lazy_import("json", "NonExistentClass")
        assert callable(result)
        with pytest.raises(ImportError):
            result()
