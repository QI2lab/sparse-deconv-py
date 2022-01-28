from setuptools import setup, find_packages

required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 'dask',
                 #'cupy', # better to instally cupy manually for your version of cuda
                 'cucim @ git+https://github.com/rapidsai/cucim.git@v21.08.01#egg=cucim&subdirectory=python/cucim']
# for installing cucim on windows, see this discussion https://github.com/rapidsai/cucim/issues/86
# the above command install only the cucim.scikit-image from a certain branch on their git repo

setup(
    name='sparse_recon',
    version='0.0.0',
    description="",
    author="",
    packages=find_packages(include=["sparse_recon"]),
    python_requires='>=3.7',
    install_requires=required_pkgs)
