"""
Setup configuration for fraud detection package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraud-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MLOps project for credit card fraud detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-fraud-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "pylint>=3.0.3",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-train=models.train:main",
            "fraud-predict=models.predict:main",
            "fraud-monitor=monitoring.monitor:main",
        ],
    },
)