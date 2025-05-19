from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flameback-financial-ratios",
    version="0.1.0",
    author="Flameback Capital Pvt Ltd",
    description="A comprehensive financial ratios analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshmin-flameback/FlamebackFinancialRatios.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)
