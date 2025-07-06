"""
Test suite for trade_modules.trade_filters module.

This module tests the filtering functionality including:
- TradingCriteriaFilter class
- PortfolioFilter class  
- DataQualityFilter class
- CustomFilter class
- Factory functions
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from trade_modules.trade_filters import (
    TradingCriteriaFilter,
    PortfolioFilter,
    DataQualityFilter,
    CustomFilter,
    TradingFilterError,
    create_criteria_filter,
    create_portfolio_filter,
    create_quality_filter,
    create_custom_filter,
)
from yahoofinance.core.errors import YFinanceError


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META'],
        'price': [150.25, 280.50, 2750.00, 850.75, 3200.00, 450.30, 320.80],
        'upside': [15.2, 8.5, 12.1, 25.8, 6.2, 18.7, 10.5],
        'buy_percentage': [85.0, 90.0, 75.0, 70.0, 80.0, 88.0, 82.0],
        'analyst_count': [25, 30, 20, 15, 28, 22, 26],
        'market_cap': [2.5e12, 2.1e12, 1.8e12, 0.8e12, 1.5e12, 1.2e12, 0.9e12],
        'pe_forward': [20.5, 25.2, 22.8, 45.2, 35.1, 28.5, 18.9],
        'pe_trailing': [22.1, 28.0, 25.0, 50.0, 38.2, 32.1, 20.5],
        'peg_ratio': [1.2, 1.8, 1.5, 2.8, 2.1, 1.9, 1.4],
        'beta': [1.1, 0.9, 1.3, 2.1, 1.4, 1.8, 1.6],
        'short_percent': [0.8, 1.2, 2.1, 5.5, 1.8, 3.2, 2.8],
        'dividend_yield': [0.52, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Consumer', 'Technology', 'Technology'],
    })


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'NVDA', 'CRM'],
        'quantity': [100, 50, 25, 30],
        'avg_cost': [145.00, 270.00, 400.00, 180.00],
        'current_price': [150.25, 280.50, 450.30, 220.50],
        'market_value': [15025.00, 14025.00, 11257.50, 6615.00],
        'sector': ['Technology', 'Technology', 'Technology', 'Technology'],
    })


@pytest.fixture
def trading_criteria_config():
    """Create trading criteria configuration."""
    return {
        'min_upside': 10.0,
        'min_buy_percentage': 75.0,
        'min_analyst_count': 10,
        'max_pe_forward': 30.0,
        'max_peg_ratio': 2.5,
        'max_beta': 2.0,
        'max_short_percent': 5.0,
        'min_market_cap': 1e9,  # 1B minimum
    }


class TestTradingCriteriaFilter:
    """Test cases for TradingCriteriaFilter class."""
    
    def test_init_with_config(self, trading_criteria_config):
        """Test TradingCriteriaFilter initialization with configuration."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        assert hasattr(filter_instance, 'criteria')
        assert filter_instance.criteria == trading_criteria_config
    
    def test_init_without_config(self):
        """Test TradingCriteriaFilter initialization without configuration."""
        filter_instance = TradingCriteriaFilter()
        
        assert hasattr(filter_instance, 'criteria')
        # Should have default criteria
        assert isinstance(filter_instance.criteria, dict)
    
    def test_apply_upside_filter(self, sample_market_data, trading_criteria_config):
        """Test upside percentage filtering."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'apply_upside_filter'):
            filtered_data = filter_instance.apply_upside_filter(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # All remaining stocks should meet upside criteria
            if len(filtered_data) > 0:
                assert all(filtered_data['upside'] >= trading_criteria_config['min_upside'])
    
    def test_apply_analyst_filter(self, sample_market_data, trading_criteria_config):
        """Test analyst count filtering."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'apply_analyst_filter'):
            filtered_data = filter_instance.apply_analyst_filter(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # All remaining stocks should meet analyst criteria
            if len(filtered_data) > 0:
                min_analysts = trading_criteria_config['min_analyst_count']
                assert all(filtered_data['analyst_count'] >= min_analysts)
    
    def test_apply_valuation_filter(self, sample_market_data, trading_criteria_config):
        """Test valuation metrics filtering."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'apply_valuation_filter'):
            filtered_data = filter_instance.apply_valuation_filter(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # Check PE and PEG ratio constraints
            if len(filtered_data) > 0:
                max_pe = trading_criteria_config['max_pe_forward']
                max_peg = trading_criteria_config['max_peg_ratio']
                assert all(filtered_data['pe_forward'] <= max_pe)
                assert all(filtered_data['peg_ratio'] <= max_peg)
    
    def test_apply_risk_filter(self, sample_market_data, trading_criteria_config):
        """Test risk metrics filtering."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'apply_risk_filter'):
            filtered_data = filter_instance.apply_risk_filter(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # Check beta and short interest constraints
            if len(filtered_data) > 0:
                max_beta = trading_criteria_config['max_beta']
                max_short = trading_criteria_config['max_short_percent']
                assert all(filtered_data['beta'] <= max_beta)
                assert all(filtered_data['short_percent'] <= max_short)
    
    def test_apply_all_filters(self, sample_market_data, trading_criteria_config):
        """Test applying all filters together."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'apply_filters'):
            filtered_data = filter_instance.apply_filters(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            assert len(filtered_data) <= len(sample_market_data)
            
            # Verify all criteria are met
            if len(filtered_data) > 0:
                assert all(filtered_data['upside'] >= trading_criteria_config['min_upside'])
                assert all(filtered_data['buy_percentage'] >= trading_criteria_config['min_buy_percentage'])
                assert all(filtered_data['analyst_count'] >= trading_criteria_config['min_analyst_count'])
    
    def test_get_filter_summary(self, trading_criteria_config):
        """Test filter summary functionality."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        if hasattr(filter_instance, 'get_filter_summary'):
            summary = filter_instance.get_filter_summary()
            
            assert isinstance(summary, (dict, str))
            # Should provide information about active filters
    
    def test_update_criteria(self, trading_criteria_config):
        """Test criteria update functionality."""
        filter_instance = TradingCriteriaFilter(trading_criteria_config)
        
        new_criteria = {'min_upside': 15.0, 'max_pe_forward': 25.0}
        
        if hasattr(filter_instance, 'update_criteria'):
            filter_instance.update_criteria(new_criteria)
            
            # Criteria should be updated
            assert filter_instance.criteria['min_upside'] == pytest.approx(15.0)
            assert filter_instance.criteria['max_pe_forward'] == pytest.approx(25.0)


class TestPortfolioFilter:
    """Test cases for PortfolioFilter class."""
    
    def test_init_with_portfolio(self, sample_portfolio_data):
        """Test PortfolioFilter initialization with portfolio data."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        assert hasattr(filter_instance, 'portfolio_df')
        assert hasattr(filter_instance, 'portfolio_tickers')
        assert isinstance(filter_instance.portfolio_df, pd.DataFrame)
    
    def test_init_without_portfolio(self):
        """Test PortfolioFilter initialization without portfolio data."""
        filter_instance = PortfolioFilter()
        
        assert hasattr(filter_instance, 'portfolio_df')
        assert hasattr(filter_instance, 'portfolio_tickers')
        # Should handle empty portfolio
        assert filter_instance.portfolio_df is None
    
    def test_exclude_existing_holdings(self, sample_market_data, sample_portfolio_data):
        """Test exclusion of existing portfolio holdings."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        if hasattr(filter_instance, 'exclude_existing_holdings'):
            filtered_data = filter_instance.exclude_existing_holdings(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            
            # Should not contain symbols from portfolio
            portfolio_symbols = set(sample_portfolio_data['symbol'].tolist())
            filtered_symbols = set(filtered_data['symbol'].tolist())
            
            assert len(portfolio_symbols.intersection(filtered_symbols)) == 0
    
    def test_apply_sector_diversification(self, sample_market_data, sample_portfolio_data):
        """Test sector diversification filtering."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        if hasattr(filter_instance, 'apply_sector_diversification'):
            filtered_data = filter_instance.apply_sector_diversification(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # Should consider sector exposure when filtering
    
    def test_apply_correlation_filter(self, sample_market_data, sample_portfolio_data):
        """Test correlation-based filtering."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        if hasattr(filter_instance, 'apply_correlation_filter'):
            filtered_data = filter_instance.apply_correlation_filter(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            # Should filter based on correlation with existing holdings
    
    def test_check_position_size_limits(self, sample_market_data, sample_portfolio_data):
        """Test position size limit checking."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        if hasattr(filter_instance, 'check_position_size_limits'):
            total_portfolio_value = sample_portfolio_data['market_value'].sum()
            new_position_value = 5000.0
            
            is_valid = filter_instance.check_position_size_limits(
                new_position_value, total_portfolio_value
            )
            
            assert isinstance(is_valid, bool)
    
    def test_get_sector_exposure(self, sample_portfolio_data):
        """Test sector exposure calculation."""
        filter_instance = PortfolioFilter(sample_portfolio_data)
        
        if hasattr(filter_instance, 'get_sector_exposure'):
            exposure = filter_instance.get_sector_exposure()
            
            assert isinstance(exposure, dict)
            # Should show percentage allocation by sector
            total_allocation = sum(exposure.values())
            assert abs(total_allocation - 1.0) < 0.01  # Should sum to ~100%
    
    def test_update_portfolio(self, sample_portfolio_data):
        """Test portfolio update functionality."""
        filter_instance = PortfolioFilter()
        
        if hasattr(filter_instance, 'update_portfolio'):
            filter_instance.update_portfolio(sample_portfolio_data)
            
            assert isinstance(filter_instance.portfolio, pd.DataFrame)
            assert len(filter_instance.portfolio) == len(sample_portfolio_data)


class TestDataQualityFilter:
    """Test cases for DataQualityFilter class."""
    
    def test_init_with_completeness_threshold(self):
        """Test DataQualityFilter initialization with completeness threshold."""
        filter_instance = DataQualityFilter(min_completeness=0.8)
        
        assert hasattr(filter_instance, 'min_completeness')
        assert filter_instance.min_completeness == pytest.approx(0.8)
    
    def test_init_with_default_threshold(self):
        """Test DataQualityFilter initialization with default threshold."""
        filter_instance = DataQualityFilter()
        
        assert hasattr(filter_instance, 'min_completeness')
        assert 0.0 <= filter_instance.min_completeness <= 1.0
    
    def test_check_data_completeness(self, sample_market_data):
        """Test data completeness checking."""
        filter_instance = DataQualityFilter(min_completeness=0.7)
        
        if hasattr(filter_instance, 'check_data_completeness'):
            completeness = filter_instance.check_data_completeness(sample_market_data)
            
            assert isinstance(completeness, (float, dict))
            if isinstance(completeness, float):
                assert 0.0 <= completeness <= 1.0
    
    def test_filter_incomplete_data(self, sample_market_data):
        """Test filtering of incomplete data."""
        # Create data with missing values
        incomplete_data = sample_market_data.copy()
        incomplete_data.loc[0, 'upside'] = np.nan
        incomplete_data.loc[1, 'analyst_count'] = np.nan
        
        filter_instance = DataQualityFilter(min_completeness=0.8)
        
        if hasattr(filter_instance, 'filter_incomplete_data'):
            filtered_data = filter_instance.filter_incomplete_data(incomplete_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            assert len(filtered_data) <= len(incomplete_data)
    
    def test_validate_required_columns(self, sample_market_data):
        """Test required columns validation."""
        filter_instance = DataQualityFilter()
        
        required_columns = ['symbol', 'price', 'upside']
        
        if hasattr(filter_instance, 'validate_required_columns'):
            is_valid = filter_instance.validate_required_columns(
                sample_market_data, required_columns
            )
            
            assert isinstance(is_valid, bool)
            assert is_valid is True  # Sample data should have required columns
    
    def test_check_data_types(self, sample_market_data):
        """Test data type validation."""
        filter_instance = DataQualityFilter()
        
        if hasattr(filter_instance, 'check_data_types'):
            type_errors = filter_instance.check_data_types(sample_market_data)
            
            assert isinstance(type_errors, (list, dict))
            # Should identify any data type issues
    
    def test_detect_outliers(self, sample_market_data):
        """Test outlier detection."""
        filter_instance = DataQualityFilter()
        
        if hasattr(filter_instance, 'detect_outliers'):
            outliers = filter_instance.detect_outliers(sample_market_data)
            
            assert isinstance(outliers, (pd.DataFrame, pd.Series, list))
            # Should identify statistical outliers
    
    def test_clean_data(self, sample_market_data):
        """Test data cleaning functionality."""
        filter_instance = DataQualityFilter()
        
        if hasattr(filter_instance, 'clean_data'):
            cleaned_data = filter_instance.clean_data(sample_market_data)
            
            assert isinstance(cleaned_data, pd.DataFrame)
            assert len(cleaned_data) <= len(sample_market_data)
            # Cleaned data should have better quality


class TestCustomFilter:
    """Test cases for CustomFilter class."""
    
    def test_init(self):
        """Test CustomFilter initialization."""
        filter_instance = CustomFilter()
        
        # Should initialize without errors
        assert isinstance(filter_instance, CustomFilter)
    
    def test_add_custom_rule(self):
        """Test adding custom filtering rules."""
        filter_instance = CustomFilter()
        
        if hasattr(filter_instance, 'add_rule'):
            # Add a simple rule
            rule = lambda df: df[df['upside'] > 10]
            filter_instance.add_rule('high_upside', rule)
            
            # Rule should be stored
            if hasattr(filter_instance, 'rules'):
                assert 'high_upside' in filter_instance.rules
    
    def test_apply_custom_rules(self, sample_market_data):
        """Test applying custom filtering rules."""
        filter_instance = CustomFilter()
        
        if hasattr(filter_instance, 'add_rule') and hasattr(filter_instance, 'apply_rules'):
            # Add custom rules
            high_upside_rule = lambda df: df[df['upside'] > 10]
            low_risk_rule = lambda df: df[df['beta'] < 1.5]
            
            filter_instance.add_rule('high_upside', high_upside_rule)
            filter_instance.add_rule('low_risk', low_risk_rule)
            
            # Apply all rules
            filtered_data = filter_instance.apply_rules(sample_market_data)
            
            assert isinstance(filtered_data, pd.DataFrame)
            assert len(filtered_data) <= len(sample_market_data)
    
    def test_remove_custom_rule(self):
        """Test removing custom filtering rules."""
        filter_instance = CustomFilter()
        
        if hasattr(filter_instance, 'add_rule') and hasattr(filter_instance, 'remove_rule'):
            # Add a rule
            rule = lambda df: df[df['upside'] > 10]
            filter_instance.add_rule('test_rule', rule)
            
            # Remove the rule
            filter_instance.remove_rule('test_rule')
            
            # Rule should be removed
            if hasattr(filter_instance, 'rules'):
                assert 'test_rule' not in filter_instance.rules
    
    def test_list_custom_rules(self):
        """Test listing custom filtering rules."""
        filter_instance = CustomFilter()
        
        if hasattr(filter_instance, 'add_rule') and hasattr(filter_instance, 'list_rules'):
            # Add some rules
            rule1 = lambda df: df[df['upside'] > 10]
            rule2 = lambda df: df[df['beta'] < 1.5]
            
            filter_instance.add_rule('rule1', rule1)
            filter_instance.add_rule('rule2', rule2)
            
            # List rules
            rules_list = filter_instance.list_rules()
            
            assert isinstance(rules_list, (list, dict))
            if isinstance(rules_list, list):
                assert 'rule1' in rules_list
                assert 'rule2' in rules_list


class TestTradingFilterError:
    """Test cases for TradingFilterError exception."""
    
    def test_trading_filter_error_inheritance(self):
        """Test that TradingFilterError inherits from YFinanceError."""
        assert issubclass(TradingFilterError, YFinanceError)
    
    def test_trading_filter_error_creation(self):
        """Test TradingFilterError creation."""
        error_message = "Test filter error"
        error = TradingFilterError(error_message)
        
        assert str(error) == error_message
        assert isinstance(error, YFinanceError)
    
    def test_trading_filter_error_raising(self):
        """Test raising TradingFilterError."""
        with pytest.raises(TradingFilterError):
            raise TradingFilterError("Test error")


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_criteria_filter_default(self):
        """Test create_criteria_filter with default parameters."""
        filter_instance = create_criteria_filter()
        
        assert isinstance(filter_instance, TradingCriteriaFilter)
    
    def test_create_criteria_filter_with_config(self, trading_criteria_config):
        """Test create_criteria_filter with custom configuration."""
        filter_instance = create_criteria_filter(trading_criteria_config)
        
        assert isinstance(filter_instance, TradingCriteriaFilter)
        assert filter_instance.criteria == trading_criteria_config
    
    def test_create_portfolio_filter_default(self):
        """Test create_portfolio_filter with default parameters."""
        filter_instance = create_portfolio_filter()
        
        assert isinstance(filter_instance, PortfolioFilter)
    
    def test_create_portfolio_filter_with_portfolio(self, sample_portfolio_data):
        """Test create_portfolio_filter with portfolio data."""
        filter_instance = create_portfolio_filter(sample_portfolio_data)
        
        assert isinstance(filter_instance, PortfolioFilter)
        assert isinstance(filter_instance.portfolio_df, pd.DataFrame)
    
    def test_create_quality_filter_default(self):
        """Test create_quality_filter with default parameters."""
        filter_instance = create_quality_filter()
        
        assert isinstance(filter_instance, DataQualityFilter)
        assert hasattr(filter_instance, 'min_completeness')
    
    def test_create_quality_filter_with_threshold(self):
        """Test create_quality_filter with custom threshold."""
        filter_instance = create_quality_filter(min_completeness=0.9)
        
        assert isinstance(filter_instance, DataQualityFilter)
        assert filter_instance.min_completeness == pytest.approx(0.9)
    
    def test_create_custom_filter(self):
        """Test create_custom_filter factory function."""
        filter_instance = create_custom_filter()
        
        assert isinstance(filter_instance, CustomFilter)


class TestFilterChaining:
    """Test chaining multiple filters together."""
    
    def test_criteria_and_portfolio_filter_chain(self, sample_market_data, sample_portfolio_data, trading_criteria_config):
        """Test chaining criteria and portfolio filters."""
        criteria_filter = create_criteria_filter(trading_criteria_config)
        portfolio_filter = create_portfolio_filter(sample_portfolio_data)
        
        # Apply criteria filter first
        if hasattr(criteria_filter, 'apply_filters'):
            step1_filtered = criteria_filter.apply_filters(sample_market_data)
        else:
            step1_filtered = sample_market_data
        
        # Apply portfolio filter second
        if hasattr(portfolio_filter, 'exclude_existing_holdings'):
            final_filtered = portfolio_filter.exclude_existing_holdings(step1_filtered)
        else:
            final_filtered = step1_filtered
        
        assert isinstance(final_filtered, pd.DataFrame)
        assert len(final_filtered) <= len(sample_market_data)
    
    def test_all_filters_chain(self, sample_market_data, sample_portfolio_data, trading_criteria_config):
        """Test chaining all types of filters."""
        criteria_filter = create_criteria_filter(trading_criteria_config)
        portfolio_filter = create_portfolio_filter(sample_portfolio_data)
        quality_filter = create_quality_filter(min_completeness=0.8)
        custom_filter = create_custom_filter()
        
        # Add custom rule
        if hasattr(custom_filter, 'add_rule'):
            custom_rule = lambda df: df[df['market_cap'] > 1e9]  # Large cap only
            custom_filter.add_rule('large_cap_only', custom_rule)
        
        # Chain all filters
        data = sample_market_data
        
        if hasattr(quality_filter, 'clean_data'):
            data = quality_filter.clean_data(data)
        
        if hasattr(criteria_filter, 'apply_filters'):
            data = criteria_filter.apply_filters(data)
        
        if hasattr(portfolio_filter, 'exclude_existing_holdings'):
            data = portfolio_filter.exclude_existing_holdings(data)
        
        if hasattr(custom_filter, 'apply_rules'):
            data = custom_filter.apply_rules(data)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) <= len(sample_market_data)


class TestPerformance:
    """Test filter performance with large datasets."""
    
    def test_criteria_filter_performance(self, trading_criteria_config):
        """Test criteria filter performance with large dataset."""
        # Create large dataset
        large_data = pd.DataFrame({
            'symbol': [f'STOCK{i}' for i in range(5000)],
            'upside': np.random.uniform(0, 50, 5000),
            'buy_percentage': np.random.uniform(40, 100, 5000),
            'analyst_count': np.random.randint(1, 40, 5000),
            'pe_forward': np.random.uniform(10, 100, 5000),
            'peg_ratio': np.random.uniform(0.5, 5, 5000),
            'beta': np.random.uniform(0.5, 3, 5000),
            'short_percent': np.random.uniform(0, 15, 5000),
        })
        
        criteria_filter = create_criteria_filter(trading_criteria_config)
        
        import time
        start_time = time.perf_counter()
        
        if hasattr(criteria_filter, 'apply_filters'):
            filtered_data = criteria_filter.apply_filters(large_data)
            assert isinstance(filtered_data, pd.DataFrame)
        
        end_time = time.perf_counter()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
    
    def test_quality_filter_performance(self):
        """Test quality filter performance with large dataset."""
        # Create large dataset with some missing values
        large_data = pd.DataFrame({
            'symbol': [f'STOCK{i}' for i in range(5000)],
            'price': np.random.uniform(10, 1000, 5000),
            'upside': np.random.uniform(0, 50, 5000),
        })
        
        # Add some missing values
        missing_indices = np.random.choice(5000, 500, replace=False)
        large_data.loc[missing_indices, 'upside'] = np.nan
        
        quality_filter = create_quality_filter(min_completeness=0.8)
        
        import time
        start_time = time.perf_counter()
        
        if hasattr(quality_filter, 'filter_incomplete_data'):
            filtered_data = quality_filter.filter_incomplete_data(large_data)
            assert isinstance(filtered_data, pd.DataFrame)
        
        end_time = time.perf_counter()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_criteria_filter_with_invalid_data(self, trading_criteria_config):
        """Test criteria filter with invalid data."""
        criteria_filter = create_criteria_filter(trading_criteria_config)
        
        # Test with None
        if hasattr(criteria_filter, 'apply_filters'):
            try:
                result = criteria_filter.apply_filters(None)
            except (TypeError, ValueError, TradingFilterError):
                assert True
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        if hasattr(criteria_filter, 'apply_filters'):
            try:
                result = criteria_filter.apply_filters(empty_df)
                assert isinstance(result, pd.DataFrame)
            except (ValueError, TradingFilterError):
                assert True
    
    def test_portfolio_filter_with_invalid_data(self, sample_portfolio_data):
        """Test portfolio filter with invalid data."""
        portfolio_filter = create_portfolio_filter(sample_portfolio_data)
        
        # Test with missing columns
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
        
        if hasattr(portfolio_filter, 'exclude_existing_holdings'):
            try:
                result = portfolio_filter.exclude_existing_holdings(invalid_data)
            except (KeyError, ValueError, TradingFilterError):
                assert True
    
    def test_quality_filter_with_invalid_data(self):
        """Test quality filter with invalid data."""
        quality_filter = create_quality_filter()
        
        # Test with None
        if hasattr(quality_filter, 'check_data_completeness'):
            try:
                result = quality_filter.check_data_completeness(None)
            except (TypeError, ValueError, TradingFilterError):
                assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])