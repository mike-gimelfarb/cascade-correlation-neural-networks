import pathlib
from setuptools import setup

# set up information
setup(
    name="cascor",
    version="0.0.1",
    description="A simple package for building and training constructive feed-forward neural networks based on tensorflow.",
    url="https://github.com/mike-gimelfarb/cascade-correlation-neural-networks",
    author="Michael Gimelfarb",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["cascor"],
    include_package_data=True,
    install_requires=["numpy>=1.19.2",
                      "scipy>=1.6.2",
                      "pandas>=1.2.4",
                      "seaborn>=0.11.1",
                      "matplotlib>=3.3.4"],
                      #"tensorflow>=2.3.1"],
    entry_points={
        
    },
)
