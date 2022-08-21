from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}
ext_modules = []
if CUDA_HOME is None:
    raise NotImplementedError('The torch_knnquery module only works with CUDA')

ext_modules += [
    CUDAExtension('knnquery_cuda',
                    ['cuda/knnquery.cpp', 'cuda/knnquery.cu'])
]
__version__ = '1.0.0'
#url = 'https://github.com/janericlenssen/torch_knnquery'

install_requires = ['torchvision']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'numpy']

setup(
    name='torch_knnquery',
    version=__version__,
    description='Implementation of voxel grid data structure for KNN point queries on the GPU in PyTorch',
    author='Jan Eric Lenssen',
    author_email='janeric.lenssen@tu-dortmund.de',
    #url=url,
    #download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch', 'voxel grid', 'knn', 'neural points'
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)