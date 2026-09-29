"""
Setup configuration for comparative_forecasting_study package.

This package implements a comparative time-series forecasting study comparing
DLinear, NLinear, PatchTST, and iTransformer models on smart grid load forecasting.
"""

from setuptools import setup, find_packages

setup(
    name="comparative_forecasting_study",
    version="0.1.0",
    description="Comparative Analysis of Transformers and Linear Models for Electricity Consumption Forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="thesis@example.com",
    url="https://github.com/example/comparative-forecasting-study",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "torch==2.0.1",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "pyyaml==6.0",
        "statsmodels==0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cfs-audit=experiments.audit:main",
            "cfs-run=experiments.run:main",
            "cfs-report=experiments.report:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="time-series forecasting electricity load prediction transformers deep-learning",
)
