from pkg_resources import DistributionNotFound, get_distribution
import pathlib
from setuptools import setup


# check for a working tensorflow version
def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


tf_deps = []
if get_dist('tensorflow') is None and get_dist('tensorflow-gpu') is None:
    tf_deps = ['tensorflow']
     
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
    packages=["cascor", "examples"],
    include_package_data=True,
    install_requires=["numpy>=1.19.2",
                      "scipy>=1.6.2",
                      "pandas>=1.2.4",
                      "seaborn>=0.11.1",
                      "matplotlib>=3.3.4"] + tf_deps,
    entry_points={
        
    },
)
