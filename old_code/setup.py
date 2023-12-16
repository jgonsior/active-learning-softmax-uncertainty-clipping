import json

from pathlib import Path
from setuptools import setup, find_packages

with open(Path(__file__).parent.joinpath("small_text/version.json")) as f:
    version = json.load(f)

version_str = ".".join(map(str, [version["major"], version["minor"], version["micro"]]))
if version["pre_release"] != "":
    version_str += "." + version["pre_release"]


PYTORCH_DEPENDENCIES = ["torch>=1.6.0", "torchtext>=0.7.0"]


setup(
    name="small-text",
    version=version_str,
    license="MIT License",
    description="A simple, modular active learning library for text classification.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Christopher Schröder",
    author_email="small-text@protonmail.com",
    url="https://github.com/webis-de/small-text",
    keywords=["active learning", "text classification"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["dill", "scipy", "numpy>=1.20.0", "scikit-learn>=0.24.1", "tqdm"],
    extras_require={
        "pytorch": PYTORCH_DEPENDENCIES,
        "transformers": PYTORCH_DEPENDENCIES + ["transformers>=4.0.0"],
    },
)
