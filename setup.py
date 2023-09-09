import pathlib

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

LIBRARY_NAME = "quick_torch"  # Rename according to te "library" folder

# List of requirements
with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

# Long description
with pathlib.Path("README.md") as f:
    long_description = f.read()

setup(
    name=LIBRARY_NAME,
    packages=find_packages(include=[LIBRARY_NAME]),
    version="1.0.0",
    description=('Library that provides a QuickDraw dataset using the Pytorch API.'),
    # long_description=long_description,
    author="Francisco Munoz G.",
    license="MIT",
    install_requires=install_requires,
    # python_requires='>=3.10',
    setup_requires=["pytest-runner"],
    tests_requires=["pytest==4.4.1"],
    test_suite="tests",
)
