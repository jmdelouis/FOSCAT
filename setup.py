from setuptools import setup, find_packages


setup(
    name='foscat',
    version='2.3.4',
    description='Generate synthetic Healpix or 2D data using Cross Scattering Transform' ,
    long_description='Utilize the Cross Scattering Transform (described in https://arxiv.org/abs/2207.12527) to synthesize Healpix or 2D data that is suitable for component separation purposes, such as denoising. \n A demo package for this process can be found at https://github.com/jmdelouis/FOSCAT_DEMO. \n\n List of developers : J.-M. Delouis, T. Foulquier, L. Mousset, E. Allys ' ,
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
