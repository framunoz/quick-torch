from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

# import quick_torch

LIBRARY_NAME = "quick_torch"  # Rename according to te "library" folder

# List of requirements
# install_requires = [
#     requirement.strip() for requirement in open("requirements.txt").readlines()
# ]
with Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

# Long description
long_description = Path("README.md").read_text()

setup(
    name=LIBRARY_NAME,
    packages=find_packages(include=[LIBRARY_NAME]),
    version="1.0.1",
    description=('Library that provides a QuickDraw dataset using the Pytorch API.'),
    url='https://github.com/framunoz/quick-torch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Francisco Muñoz G.",
    license="MIT",
    license_files=["LICENSE"],
    install_requires=install_requires,
    python_requires='>=3.10',
    setup_requires=["pytest-runner"],
    tests_requires=["pytest==4.4.1"],
    test_suite="tests",
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA :: 11.8',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Visualization',
    ]
)
