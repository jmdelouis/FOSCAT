from setuptools import setup, find_packages


setup(
    name='foscat',
    version='0.1',
    license='MIT',
    author="Jean-Marc DELOUIS",
    author_email='jean.marc.delouis@ifremer.fr',
    packages=find_packages('src'),
    package_dir={'': 'src' : 'demo'},
    url='https://gitlab.ifremer.fr/deepsee/foscat',
    keywords='Scattering transform',
    install_requires=[
          'sys',
          'os',
          'imageio',
          'matplotlib',
          'numpy',
          'tensorflow-gpu',
          'healpy',
      ],

)
