#!/usr/bin/env python
"""
Setup script for the etorotrade package.
"""

import os
import re
from setuptools import setup, find_packages


# Read the version from yahoofinance/__init__.py
with open(os.path.join("yahoofinance", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="etorotrade",
    version=version,
    description="Market analysis and portfolio management tool using Yahoo Finance data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Roo",
    author_email="noreply@example.com",
    url="https://github.com/weirdapps/etorotrade",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "flake8-bugbear>=23.5.9",
            "flake8-bandit>=4.1.1",
            "mypy>=1.3.0",
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pre-commit>=3.3.2",
        ],
        "docs": [
            "sphinx>=7.0.1",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "etorotrade=trade:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="finance, trading, stock market, portfolio management, yahoo finance",
    project_urls={
        "Bug Reports": "https://github.com/weirdapps/etorotrade/issues",
        "Source": "https://github.com/weirdapps/etorotrade",
    },
)