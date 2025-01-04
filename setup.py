from pkg_resources import DistributionNotFound, get_distribution
import pathlib
from setuptools import setup, find_packages


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
    name="pyccnn",
    version="0.1",
    author="Michael Gimelfarb",
    author_email="mike.gimelfarb@mail.utoronto.ca",
    description="A package for building and training cascade correlation neural networks in tensorflow and scipy.",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/mike-gimelfarb/cascade-correlation-neural-networks",    
    packages=find_packages(),
    package_data={'': ['*.csv']},
    include_package_data=True,
    install_requires=["numpy>=1.19.2",
                      "scipy>=1.6.2",
                      "scikit-learn",
                      "pandas>=1.2.4",
                      "seaborn>=0.11.1",
                      "matplotlib>=3.3.4",
                      "tqdm"] + tf_deps,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
