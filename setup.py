"""Setup script for Seedcraft."""

from setuptools import setup, find_packages

setup(
    name="seedcraft",
    version="0.1.0",
    description="Deterministic, correlation-aware synthetic data generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Seedcraft Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "seedcraft=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Libraries",
    ],
)
