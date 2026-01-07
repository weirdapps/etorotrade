"""
Tests for yahoofinance/utils/network/batch.py

This module tests batch processing utilities.
"""

import pytest
from unittest.mock import MagicMock, patch

from yahoofinance.utils.network.batch import BatchProcessor, batch_process


class TestBatchProcessorInit:
    """Tests for BatchProcessor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        processor = BatchProcessor(process_func=lambda x: x)
        assert processor.process_func is not None
        assert processor.batch_size > 0
        assert processor.batch_delay >= 0
        assert processor.max_workers > 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        processor = BatchProcessor(
            process_func=lambda x: x * 2,
            batch_size=10,
            batch_delay=0.1,
            max_workers=5,
        )
        assert processor.batch_size == 10
        assert processor.batch_delay == pytest.approx(0.1)
        assert processor.max_workers == 5


class TestBatchProcessorCreateBatches:
    """Tests for _create_batches method."""

    def test_create_batches_even_split(self):
        """Test creating batches that split evenly."""
        processor = BatchProcessor(process_func=lambda x: x, batch_size=3)
        items = [1, 2, 3, 4, 5, 6]
        batches = processor._create_batches(items)
        assert len(batches) == 2
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]

    def test_create_batches_uneven_split(self):
        """Test creating batches that don't split evenly."""
        processor = BatchProcessor(process_func=lambda x: x, batch_size=3)
        items = [1, 2, 3, 4, 5]
        batches = processor._create_batches(items)
        assert len(batches) == 2
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5]

    def test_create_batches_single_item(self):
        """Test creating batches with a single item."""
        processor = BatchProcessor(process_func=lambda x: x, batch_size=3)
        items = [1]
        batches = processor._create_batches(items)
        assert len(batches) == 1
        assert batches[0] == [1]

    def test_create_batches_empty_list(self):
        """Test creating batches from empty list."""
        processor = BatchProcessor(process_func=lambda x: x, batch_size=3)
        items = []
        batches = processor._create_batches(items)
        assert len(batches) == 0


class TestBatchProcessorProcess:
    """Tests for process method."""

    def test_process_success(self):
        """Test processing items successfully."""
        processor = BatchProcessor(
            process_func=lambda x: x * 2,
            batch_size=3,
            batch_delay=0,
            max_workers=2,
        )
        items = [1, 2, 3, 4, 5]
        results = processor.process(items)
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]

    def test_process_empty_list(self):
        """Test processing empty list."""
        processor = BatchProcessor(process_func=lambda x: x * 2)
        results = processor.process([])
        assert len(results) == 0

    def test_process_single_item(self):
        """Test processing single item."""
        processor = BatchProcessor(
            process_func=lambda x: x + 1,
            batch_size=10,
            batch_delay=0,
        )
        results = processor.process([5])
        assert results == [6]

    def test_process_preserves_order(self):
        """Test that results preserve order."""
        processor = BatchProcessor(
            process_func=lambda x: x,
            batch_size=2,
            batch_delay=0,
        )
        items = [1, 2, 3, 4, 5]
        results = processor.process(items)
        assert results == items

    def test_process_with_complex_function(self):
        """Test processing with a complex function."""
        def complex_func(item):
            return {"value": item, "doubled": item * 2}

        processor = BatchProcessor(
            process_func=complex_func,
            batch_size=2,
            batch_delay=0,
        )
        results = processor.process([1, 2, 3])
        assert len(results) == 3
        assert results[0]["value"] == 1
        assert results[0]["doubled"] == 2


class TestBatchProcessorCancellation:
    """Tests for batch processor cancellation."""

    def test_cancel_method_exists(self):
        """Test that cancel method exists."""
        processor = BatchProcessor(process_func=lambda x: x)
        assert hasattr(processor, 'cancel')
        assert hasattr(processor, 'cancel_event')

    def test_initial_cancel_state(self):
        """Test that cancel event is not set initially."""
        processor = BatchProcessor(process_func=lambda x: x)
        assert not processor.cancel_event.is_set()

    def test_cancel_sets_event(self):
        """Test that cancel method sets the event."""
        processor = BatchProcessor(process_func=lambda x: x)
        processor.cancel()
        assert processor.cancel_event.is_set()


class TestBatchProcessorEdgeCases:
    """Edge case tests for batch processor."""

    def test_large_batch_size(self):
        """Test with batch size larger than items."""
        processor = BatchProcessor(
            process_func=lambda x: x,
            batch_size=100,
        )
        items = [1, 2, 3]
        batches = processor._create_batches(items)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        processor = BatchProcessor(
            process_func=lambda x: x,
            batch_size=1,
        )
        items = [1, 2, 3]
        batches = processor._create_batches(items)
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)


class TestBatchProcessFunction:
    """Tests for batch_process convenience function."""

    def test_batch_process_success(self):
        """Test batch_process function."""
        items = [1, 2, 3, 4, 5]
        results = batch_process(
            items=items,
            process_func=lambda x: x * 2,
            batch_size=3,
            batch_delay=0,
            max_workers=2,
        )
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]

    def test_batch_process_empty_list(self):
        """Test batch_process with empty list."""
        results = batch_process(
            items=[],
            process_func=lambda x: x * 2,
        )
        assert results == []
