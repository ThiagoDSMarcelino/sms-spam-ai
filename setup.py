"""Setup script."""

import os
import pathlib
from setuptools import setup, find_packages

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

def get_requirements():
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    
    return requirements

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = get_version("src/version.py")

setup(
    name="sms_spam_ai",
    description="A project to classify SMS messages as spam or non-spam using artificial intelligence",
    long_description_content_type="text/markdown",
    long_description=README,
    url="https://github.com/ThiagoDSMarcelino/sms-spam-ai",
    author="Thiago dos Santos Marcelino",
    author_email="thiagodsmarcelino@gmail.com",
    license="MIT",
    packages=find_packages("src"),
    entry_points={
        "console_scripts": [
            "sms_spam_ai = sms_spam_ai.main:main",
        ],
    },
    install_requires=get_requirements(),
)
