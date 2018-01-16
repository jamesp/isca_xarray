# This file allows the installation of isca_xarray utilities as a python library 'iscaxr'
# Suggested installation procedure:
# 		$ git clone https://github.com/jamesp/isca_xarray
# 		$ cd isca_xarray
#		  $ pip install -e .
# This installs the package in *development mode* i.e. any changes you make to these files
# or any additional files you add will be immediately available.
# In a new python console, from any directory, you can now use the iscaxr code:
# 		>>> import iscaxr
#		  >>> d = iscaxr.resample_latlon(...)
#

from distutils.core import setup

setup(name='iscaxr',
      version='0.1',
      description='Isca run analysis tools using xarray',
      author='James Penn',
      url='https://github.com/jamesp/iscaxr',
      packages=['iscaxr'],
      install_requires=[
        'numpy',
        'xarray',
        'scipy',
        'astropy'
      ]
     )