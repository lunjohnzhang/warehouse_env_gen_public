import sys
import os
from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
from setuptools import setup, find_packages
from glob import glob

__version__ = "0.0.1"

BOOST_INC_DIR = "/usr/include"
BOOST_LIB_DIR = "/usr/lib/x86_64-linux-gnu"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        name="warehouse_sim",
        # os.listdir("RHCR/src"),
        # ["RHCR/src/WarehouseSimulation.cpp"],
        sources=sorted(glob("RHCR/src/*.cpp")),
        include_dirs = ["RHCR/inc", BOOST_INC_DIR], # -D<string>=<string>
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)], # -D<string>=<string>
        undef_macros=[],  # [string] -D<string>
        library_dirs=[BOOST_LIB_DIR],  # [string] -L<string>
        libraries=[
            "boost_program_options",
            "boost_system",
            "boost_filesystem",
        ],  # [string] -l<string>
        runtime_library_dirs=[],  # [string] -rpath=<string>
        extra_objects=[],  # [string]
        extra_compile_args=[],  # [string]
        extra_link_args=[],  # [string]
    ),
]

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

setup(
    name="warehouse_env_gen",
    version=__version__,
    author="Yulun Zhang",
    author_email="yulunz@andrew.cmu.edu",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"cxx_std": 11},
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'torch==1.13.0+cu117',
        'torchvision==0.14.0+cu117',
        'torchaudio==0.13.0+cu117',
        'fire',
        'gin-config',
        'logdir==0.12.0',
        'ribs[all]==0.4.0',

        # Dask
        'dask==2.30.0',
        'dask-jobqueue==0.7.1',
        'distributed==2.30.0',
        'click==7.1.2',  # Newer click causes error with dask scheduler.
        'bokeh==2.2.3',
        'jupyter-server-proxy==1.5.0',

        # Plot
        'matplotlib',
        'seaborn',
        'loguru',
        'pingouin',
    ]
)
