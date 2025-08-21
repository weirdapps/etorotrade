"""
Test suite for repository pattern implementation.

This module tests the repository interfaces and implementations to ensure
they maintain backward compatibility while providing clean data access
abstraction.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from trade_modules.repositories import (
    CsvRepository,
    PortfolioRepository, 
    MarketDataRepository
)
from trade_modules.repositories.interfaces import (
    IDataRepository,
    ICsvRepository,
    IPortfolioRepository,
    IMarketDataRepository
)
from trade_modules.errors import DataProcessingError


class TestCsvRepository:
    """Test CSV repository implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def csv_repo(self, temp_dir):
        """Create CSV repository instance."""
        return CsvRepository(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'price': [100.0, 200.0, 300.0],
            'volume': [1000, 2000, 3000]
        }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    def test_repository_interface_compliance(self, csv_repo):
        """Test that CsvRepository implements required interfaces."""
        assert isinstance(csv_repo, IDataRepository)
        assert isinstance(csv_repo, ICsvRepository)
    
    def test_write_and_read_data(self, csv_repo, sample_data):
        """Test basic write and read operations."""
        # Write data
        success = csv_repo.write('test_data', sample_data)
        assert success == True
        
        # Read data back
        result_df = csv_repo.read('test_data')
        pd.testing.assert_frame_equal(result_df, sample_data)
    
    def test_exists_functionality(self, csv_repo, sample_data):
        """Test exists checking."""
        # Should not exist initially
        assert csv_repo.exists('test_data') == False
        
        # Write data
        csv_repo.write('test_data', sample_data)
        
        # Should exist now
        assert csv_repo.exists('test_data') == True
    
    def test_delete_functionality(self, csv_repo, sample_data):
        """Test delete operations."""
        # Write data
        csv_repo.write('test_data', sample_data)
        assert csv_repo.exists('test_data') == True
        
        # Delete data
        success = csv_repo.delete('test_data')
        assert success == True
        assert csv_repo.exists('test_data') == False
    
    def test_read_nonexistent_file(self, csv_repo):
        """Test reading nonexistent file returns empty DataFrame."""
        result_df = csv_repo.read('nonexistent')
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
    
    def test_get_file_path(self, csv_repo):
        """Test file path generation."""
        path = csv_repo.get_file_path('test_data')
        assert path.name == 'test_data.csv'
        assert path.parent == csv_repo.base_directory
        
        # Test with .csv extension already present
        path_with_ext = csv_repo.get_file_path('test_data.csv')
        assert path_with_ext.name == 'test_data.csv'
    
    def test_list_files(self, csv_repo, sample_data):
        """Test listing CSV files."""
        # Write multiple files
        csv_repo.write('file1', sample_data)
        csv_repo.write('file2', sample_data)
        
        files = csv_repo.list_files()
        assert len(files) >= 2
        assert any('file1.csv' in str(f) for f in files)
        assert any('file2.csv' in str(f) for f in files)
    
    def test_backup_file(self, csv_repo, sample_data):
        """Test file backup functionality."""
        # Write data
        csv_repo.write('test_data', sample_data)
        
        # Create backup
        success = csv_repo.backup_file('test_data', 'test_backup')
        assert success == True
        
        # Check backup exists
        backup_files = csv_repo.list_files('*backup*.csv')
        assert len(backup_files) >= 1
    
    def test_csv_parameters_compatibility(self, csv_repo, temp_dir):
        """Test CSV parameters maintain backward compatibility."""
        # Create test CSV file with index
        test_file = temp_dir / 'test.csv'
        sample_df = pd.DataFrame({
            'price': [100, 200],
            'volume': [1000, 2000]
        }, index=['AAPL', 'MSFT'])
        
        # Write with default parameters
        sample_df.to_csv(test_file, index=True)
        
        # Read with repository - should handle index correctly
        result_df = csv_repo.read_csv(test_file)
        assert result_df.index.tolist() == ['AAPL', 'MSFT']
        assert 'price' in result_df.columns
        assert 'volume' in result_df.columns


class TestPortfolioRepository:
    """Test portfolio repository implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def portfolio_file(self, temp_dir):
        """Create portfolio file path."""
        return temp_dir / 'portfolio.csv'
    
    @pytest.fixture
    def portfolio_repo(self, portfolio_file):
        """Create portfolio repository instance."""
        return PortfolioRepository(portfolio_file)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio DataFrame."""
        return pd.DataFrame({
            'Quantity': [100, 200, 50],
            'Value': [10000, 40000, 15000],
            'Price': [100.0, 200.0, 300.0]
        }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    def test_repository_interface_compliance(self, portfolio_repo):
        """Test that PortfolioRepository implements required interfaces."""
        assert isinstance(portfolio_repo, IDataRepository)
        assert isinstance(portfolio_repo, IPortfolioRepository)
    
    def test_get_empty_portfolio(self, portfolio_repo):
        """Test getting portfolio when file doesn't exist."""
        portfolio_df = portfolio_repo.get_portfolio()
        assert isinstance(portfolio_df, pd.DataFrame)
        assert portfolio_df.empty
    
    def test_update_and_get_portfolio(self, portfolio_repo, sample_portfolio):
        """Test updating and retrieving portfolio."""
        # Update portfolio
        success = portfolio_repo.update_portfolio(sample_portfolio)
        assert success == True
        
        # Get portfolio back
        result_df = portfolio_repo.get_portfolio()
        pd.testing.assert_frame_equal(result_df, sample_portfolio)
    
    def test_get_holdings_for_ticker(self, portfolio_repo, sample_portfolio):
        """Test getting holdings for specific ticker."""
        # Update portfolio
        portfolio_repo.update_portfolio(sample_portfolio)
        
        # Get holdings for AAPL
        holding = portfolio_repo.get_holdings_for_ticker('AAPL')
        assert holding is not None
        assert holding['Quantity'] == 100
        assert holding['Value'] == 10000
        
        # Test case insensitive
        holding_lower = portfolio_repo.get_holdings_for_ticker('aapl')
        assert holding_lower is not None
        
        # Test nonexistent ticker
        holding_none = portfolio_repo.get_holdings_for_ticker('NONEXISTENT')
        assert holding_none is None
    
    def test_add_holding(self, portfolio_repo):
        """Test adding new holding."""
        # Add holding to empty portfolio
        success = portfolio_repo.add_holding('AAPL', 100, Price=150.0, Value=15000)
        assert success == True
        
        # Verify holding was added
        holding = portfolio_repo.get_holdings_for_ticker('AAPL')
        assert holding is not None
        assert holding['Quantity'] == 100
        assert holding['Price'] == pytest.approx(150.0, 0.01)
    
    def test_remove_holding(self, portfolio_repo, sample_portfolio):
        """Test removing holding."""
        # Set up portfolio
        portfolio_repo.update_portfolio(sample_portfolio)
        
        # Remove holding
        success = portfolio_repo.remove_holding('AAPL')
        assert success == True
        
        # Verify holding was removed
        holding = portfolio_repo.get_holdings_for_ticker('AAPL')
        assert holding is None
        
        # Verify other holdings still exist
        msft_holding = portfolio_repo.get_holdings_for_ticker('MSFT')
        assert msft_holding is not None
    
    def test_portfolio_summary(self, portfolio_repo, sample_portfolio):
        """Test portfolio summary functionality."""
        # Empty portfolio summary
        empty_summary = portfolio_repo.get_portfolio_summary()
        assert empty_summary['total_holdings'] == 0
        
        # Portfolio with data
        portfolio_repo.update_portfolio(sample_portfolio)
        summary = portfolio_repo.get_portfolio_summary()
        
        assert summary['total_holdings'] == 3
        assert summary['total_value'] == 65000  # 10000 + 40000 + 15000
        assert summary['total_quantity'] == 350  # 100 + 200 + 50
        assert 'AAPL' in summary['holdings']
        assert 'MSFT' in summary['holdings']
        assert 'GOOGL' in summary['holdings']
    
    def test_portfolio_backup(self, portfolio_repo, sample_portfolio):
        """Test portfolio backup functionality."""
        # Set up portfolio
        portfolio_repo.update_portfolio(sample_portfolio)
        
        # Create backup
        success = portfolio_repo.backup_portfolio('test_backup')
        assert success == True
        
        # Verify backup exists
        backups = portfolio_repo.csv_repo.list_files('*backup*.csv')
        assert len(backups) >= 1


class TestMarketDataRepository:
    """Test market data repository implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def market_repo(self, temp_dir):
        """Create market data repository instance."""
        return MarketDataRepository(temp_dir)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market DataFrame."""
        return pd.DataFrame({
            'price': [100.0, 200.0, 300.0],
            'upside': [10.0, 15.0, 20.0],
            'confidence_score': [0.8, 0.9, 0.7]
        }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    def test_repository_interface_compliance(self, market_repo):
        """Test that MarketDataRepository implements required interfaces."""
        assert isinstance(market_repo, IDataRepository)
        assert isinstance(market_repo, IMarketDataRepository)
    
    def test_standard_data_types(self, market_repo):
        """Test that standard data types are recognized."""
        standard_types = ['buy', 'sell', 'hold', 'market']
        for data_type in standard_types:
            assert market_repo._is_valid_data_type(data_type) == True
    
    def test_save_and_get_market_data(self, market_repo, sample_market_data):
        """Test saving and retrieving market data."""
        # Save buy opportunities
        success = market_repo.save_market_data('buy', sample_market_data)
        assert success == True
        
        # Get buy opportunities back
        result_df = market_repo.get_market_data('buy')
        pd.testing.assert_frame_equal(result_df, sample_market_data)
    
    def test_get_available_data_types(self, market_repo, sample_market_data):
        """Test getting list of available data types."""
        # Initially empty
        available = market_repo.get_available_data_types()
        assert isinstance(available, list)
        
        # Save some data
        market_repo.save_market_data('buy', sample_market_data)
        market_repo.save_market_data('sell', sample_market_data)
        
        # Should now include saved types
        available = market_repo.get_available_data_types()
        assert 'buy' in available
        assert 'sell' in available
    
    def test_clear_market_data(self, market_repo, sample_market_data):
        """Test clearing market data."""
        # Save data
        market_repo.save_market_data('buy', sample_market_data)
        assert market_repo.exists('buy') == True
        
        # Clear data
        success = market_repo.clear_market_data('buy')
        assert success == True
        assert market_repo.exists('buy') == False
    
    def test_backup_market_data(self, market_repo, sample_market_data):
        """Test backing up market data."""
        # Save multiple data types
        market_repo.save_market_data('buy', sample_market_data)
        market_repo.save_market_data('sell', sample_market_data)
        
        # Backup all
        success = market_repo.backup_market_data('test_backup')
        assert success == True
        
        # Verify backups exist
        backups = market_repo.csv_repo.list_files('*backup*.csv')
        assert len(backups) >= 2
    
    def test_market_data_summary(self, market_repo, sample_market_data):
        """Test market data summary functionality."""
        # Save data
        market_repo.save_market_data('buy', sample_market_data)
        market_repo.save_market_data('sell', sample_market_data[:1])  # Smaller dataset
        
        # Get summary
        summary = market_repo.get_market_data_summary()
        
        assert 'buy' in summary
        assert 'sell' in summary
        assert summary['buy']['record_count'] == 3
        assert summary['sell']['record_count'] == 1
        assert summary['buy']['has_data'] == True
        assert 'price' in summary['buy']['columns']
    
    def test_consolidate_market_data(self, market_repo, sample_market_data):
        """Test consolidating market data."""
        # Save different data types
        buy_data = sample_market_data.copy()
        sell_data = sample_market_data.iloc[:1].copy()  # Just first row
        
        market_repo.save_market_data('buy', buy_data)
        market_repo.save_market_data('sell', sell_data)
        
        # Consolidate
        success = market_repo.consolidate_market_data('consolidated')
        assert success == True
        
        # Verify consolidated data
        consolidated_df = market_repo.get_market_data('consolidated')
        assert len(consolidated_df) == 4  # 3 buy + 1 sell
        assert 'source_type' in consolidated_df.columns
        
        # Check source tracking
        buy_records = consolidated_df[consolidated_df['source_type'] == 'buy']
        sell_records = consolidated_df[consolidated_df['source_type'] == 'sell']
        assert len(buy_records) == 3
        assert len(sell_records) == 1
    
    def test_get_nonexistent_market_data(self, market_repo):
        """Test getting nonexistent market data returns empty DataFrame."""
        result_df = market_repo.get_market_data('nonexistent')
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty


class TestRepositoryIntegration:
    """Test integration between repository components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_cross_repository_compatibility(self, temp_dir):
        """Test that repositories can work together on same directory."""
        # Create repositories sharing same base directory
        csv_repo = CsvRepository(temp_dir)
        market_repo = MarketDataRepository(temp_dir)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'price': [100, 200],
            'volume': [1000, 2000]
        }, index=['AAPL', 'MSFT'])
        
        # Save via CSV repo
        csv_repo.write('test_data', sample_data)
        
        # Read via market repo
        result_df = market_repo.get_market_data('test_data')
        pd.testing.assert_frame_equal(result_df, sample_data)
    
    def test_repository_error_handling(self, temp_dir):
        """Test error handling across repositories."""
        # Test with invalid directory permissions (if possible)
        csv_repo = CsvRepository(temp_dir)
        
        # Test with invalid data - should return False for invalid operations
        invalid_data = "not a dataframe"
        
        # Should handle gracefully and return False
        try:
            result = csv_repo.write('invalid', invalid_data)
            assert result == False
        except (TypeError, AttributeError):
            # Also acceptable to raise these exceptions
            pass
    
    def test_backward_compatibility_with_existing_patterns(self, temp_dir):
        """Test that repositories maintain backward compatibility."""
        # Create files in the old pattern
        portfolio_file = temp_dir / 'portfolio.csv'
        buy_file = temp_dir / 'buy.csv'
        
        # Create sample data using old pandas patterns
        portfolio_data = pd.DataFrame({
            'Quantity': [100, 200],
            'Value': [10000, 40000]
        }, index=['AAPL', 'MSFT'])
        
        market_data = pd.DataFrame({
            'price': [100, 200],
            'upside': [10, 15]
        }, index=['AAPL', 'MSFT'])
        
        # Save using standard pandas (old way)
        portfolio_data.to_csv(portfolio_file, index=True)
        market_data.to_csv(buy_file, index=True)
        
        # Read using repositories (new way)
        portfolio_repo = PortfolioRepository(portfolio_file)
        market_repo = MarketDataRepository(temp_dir)
        
        # Should read correctly
        portfolio_result = portfolio_repo.get_portfolio()
        market_result = market_repo.get_market_data('buy')
        
        pd.testing.assert_frame_equal(portfolio_result, portfolio_data)
        pd.testing.assert_frame_equal(market_result, market_data)


if __name__ == "__main__":
    pytest.main([__file__])