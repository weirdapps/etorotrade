#!/usr/bin/env python3
"""
ITERATION 26: Imports Utility Tests
Target: Test lazy import and dependency resolution utilities
File: yahoofinance/utils/imports.py (89 statements, 35% coverage)
"""

import pytest
from yahoofinance.utils.imports import LazyImport


class TestLazyImport:
    """Test LazyImport class."""

    def test_lazy_import_module(self):
        """Create lazy import for a module."""
        lazy_math = LazyImport('math')
        assert lazy_math._module_name == 'math'
        assert lazy_math._object_name is None

    def test_lazy_import_object(self):
        """Create lazy import for an object from a module."""
        lazy_sqrt = LazyImport('math', 'sqrt')
        assert lazy_sqrt._module_name == 'math'
        assert lazy_sqrt._object_name == 'sqrt'

    def test_lazy_import_initialization(self):
        """LazyImport starts unresolved."""
        lazy = LazyImport('os')
        assert lazy._module is None
        assert lazy._object is None

    def test_lazy_import_call_function(self):
        """Call a lazy-imported function."""
        lazy_sqrt = LazyImport('math', 'sqrt')
        result = lazy_sqrt(16)
        assert result == pytest.approx(4.0)

    def test_lazy_import_call_non_callable_raises(self):
        """Calling non-callable object raises TypeError."""
        lazy_pi = LazyImport('math', 'pi')
        with pytest.raises(TypeError, match="not callable"):
            lazy_pi()

    def test_lazy_import_getattr(self):
        """Get attribute from lazy-imported module."""
        lazy_math = LazyImport('math')
        pi = lazy_math.pi
        assert abs(pi - 3.14159) < 0.001

    def test_lazy_import_getattr_function(self):
        """Get function attribute from lazy-imported module."""
        lazy_math = LazyImport('math')
        sqrt_func = lazy_math.sqrt
        assert callable(sqrt_func)
        assert sqrt_func(9) == pytest.approx(3.0)


class TestLazyImportEdgeCases:
    """Test edge cases for LazyImport."""

    def test_lazy_import_builtin_module(self):
        """Import built-in module."""
        lazy_sys = LazyImport('sys')
        version = lazy_sys.version
        assert isinstance(version, str)

    def test_lazy_import_multiple_calls(self):
        """Multiple calls work correctly."""
        lazy_sqrt = LazyImport('math', 'sqrt')
        result1 = lazy_sqrt(16)
        result2 = lazy_sqrt(25)
        assert result1 == pytest.approx(4.0)
        assert result2 == pytest.approx(5.0)

    def test_lazy_import_with_kwargs(self):
        """Call with keyword arguments."""
        lazy_round = LazyImport('builtins', 'round')
        result = lazy_round(3.14159, ndigits=2)
        assert result == pytest.approx(3.14)

    def test_lazy_import_getattr_nested(self):
        """Access nested attributes."""
        lazy_os = LazyImport('os')
        path = lazy_os.path
        assert path is not None


class TestLazyImportInitialization:
    """Test LazyImport initialization patterns."""

    def test_module_only_initialization(self):
        """Initialize with module name only."""
        lazy = LazyImport('json')
        assert lazy._module_name == 'json'
        assert lazy._object_name is None

    def test_module_and_object_initialization(self):
        """Initialize with module and object names."""
        lazy = LazyImport('json', 'dumps')
        assert lazy._module_name == 'json'
        assert lazy._object_name == 'dumps'

    def test_private_attributes_exist(self):
        """Private attributes are initialized."""
        lazy = LazyImport('os')
        assert hasattr(lazy, '_module_name')
        assert hasattr(lazy, '_object_name')
        assert hasattr(lazy, '_module')
        assert hasattr(lazy, '_object')


class TestLazyImportCallable:
    """Test LazyImport callable behavior."""

    def test_call_with_positional_args(self):
        """Call with positional arguments."""
        lazy_max = LazyImport('builtins', 'max')
        result = lazy_max(1, 5, 3, 2, 4)
        assert result == 5

    def test_call_with_kwargs_only(self):
        """Call with keyword arguments only."""
        lazy_dict = LazyImport('builtins', 'dict')
        result = lazy_dict(a=1, b=2, c=3)
        assert result == {'a': 1, 'b': 2, 'c': 3}

    def test_call_with_mixed_args(self):
        """Call with mixed positional and keyword arguments."""
        lazy_round = LazyImport('builtins', 'round')
        result = lazy_round(3.14159, 2)
        assert result == pytest.approx(3.14)


class TestLazyImportAttributes:
    """Test LazyImport attribute access."""

    def test_getattr_simple_attribute(self):
        """Access simple module attribute."""
        lazy_math = LazyImport('math')
        e = lazy_math.e
        assert abs(e - 2.71828) < 0.001

    def test_getattr_function_attribute(self):
        """Access function from module."""
        lazy_math = LazyImport('math')
        floor = lazy_math.floor
        assert callable(floor)
        assert floor(3.7) == 3

    def test_getattr_constant(self):
        """Access constant from module."""
        lazy_math = LazyImport('math')
        tau = lazy_math.tau
        assert abs(tau - 6.28318) < 0.001


