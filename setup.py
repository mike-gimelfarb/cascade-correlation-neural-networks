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


# file information from readme
this_folder = pathlib.Path(__file__).parent
readme_file = (this_folder / "README.md").read_text()
 
# set up information
setup(
    name="cascor",
    version="0.0.1",
    description="A simple package for building and training constructive feed-forward neural networks based on tensorflow.",
    long_description=readme_file,
    long_description_content_type="text/markdown",
    url="https://github.com/mike-gimelfarb/cascade-correlation-neural-networks",
    author="Michael Gimelfarb",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    packages=["cascor", "examples"],
    include_package_data=True,
    install_requires=["numpy>=1.19.2",
                      "scipy>=1.6.2",
                      "scikit-learn",
                      "pandas>=1.2.4",
                      "seaborn>=0.11.1",
                      "matplotlib>=3.3.4",
                      "tqdm"] + tf_deps,
    entry_points={},
)
