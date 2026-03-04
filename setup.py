from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="convolutional-diffusion-models",
    version="0.1.0",
    description="Utilities and models for convolutional diffusion experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(),
    py_modules=["models", "noise_schedules"],
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.8",
)
