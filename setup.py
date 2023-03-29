from setuptools import setup, find_packages


setup(
    name='foscat',
    version='1.9.5',
    description='Synthesise 2D or Healpix data using Cross Scattering Transform' ,
    long_description='Synthesise data (2D or Healpix) using Cross Scattering Transform (https://arxiv.org/abs/2207.12527) usable for component separation (e.g. denoising). \n Demo package here : https://github.com/jmdelouis/FOSCAT_DEMO ' ,
    license='MIT',
    author='Jean-Marc DELOUIS',
    author_email='jean.marc.delouis@ifremer.fr',
    maintainer='Theo Foulquier',
    maintainer_email='theo.foulquier@ifremer.fr',
    packages=['foscat'],
    package_dir={'': 'src'},
    url='https://github.com/jmdelouis/FOSCAT',
    keywords=['Scattering transform','Component separation', 'denoising'],
    install_requires=[
          'imageio',
          'imagecodecs',
          'matplotlib',
          'numpy',
          'tensorflow',
          'healpy',
      ],

)
