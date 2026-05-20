import os
import sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'jupyter>=1.0',
    'scipy>=1.5',
    'scikit-learn>=0.23.1',
    'scikit-image>=0.17.2',
    'opencv-python>=4.3.0',
    'matplotlib>=3.3.0',
    'Cython>=0.29.22',
    'yacs>=0.1.8',
    'h5py>=2.10.0',
    'gputil>=1.4.0',
    'imageio>=2.9.0',
    'tensorboard>=2.2.2',
    'einops>=0.3.0',
    'tqdm>=4.58.0',
    'monai>=0.9.1',
    'tifffile>=2021.7.2',
    'openpyxl>=3.0',
    'gudhi>=3.7',
    'cripser>=0.3',
    'pot>=0.9',
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName)]


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/seo0229/topo3d-mito'

    setup(name='topo3d-mito',
          description='Weakly-supervised topology-preserving 3D mitochondria segmentation (built on PyTorch Connectomics)',
          version=__version__,
          url=url,
          license='MIT',
          author='PyTorch Connectomics Contributors',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
