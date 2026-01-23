# foscat

[![Read the Docs](https://readthedocs.org/projects/foscat-documentation/badge/?version=latest)](https://foscat-documentation.readthedocs.io/en/latest)

A python package dedicated to image component separation based on scattering transform analysis designed for high performance computing.

## the concept

The foscat genesis has been built to synthesise data (2D or Healpix) using Cross Scattering Transform. For a detailed method description please refer to https://arxiv.org/abs/2207.12527. This algorithm could be effectively usable for component separation (e.g. denoising).

A demo package for this process can be found at https://github.com/jmdelouis/FOSCAT_DEMO.

## usage

# Short tutorial

https://github.com/IAOCEA/demo-foscat-pangeo-eosc/blob/main/Demo_Synthesis.ipynb

# FOSCAT_DEMO

The python scripts _demo.py_ included in this package demonstrate how to use the foscat library to generate synthetic fields that have patterns with the same statistical properties as a specified image.

# Install foscat library

Before installing, make sure you have python installed in your enviroment.
The last version of the foscat library can be installed using PyPi:

```
pip install foscat
```

Load the FOSCAT_DEMO package from github.

```
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git
```

## Recommended installing procedures for mac users

It is recomended to use python=3.9\*.

```
micromamba create -n FOSCAT
micromamba install -n FOSCAT ‘python==3.9*’
micromamba activate FOSCAT
pip install foscat
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git

```

## Recommended installing procedures HPC users

It is recomended to install tensorflow in advance. For [DATARMOR](https://pcdm.ifremer.fr/Equipement) for using GPU ;

```
micromamba create -n FOSCAT
micromamba install -n FOSCAT ‘python==3.9*’
micromamba install -n FOSCAT ‘tensorflow==2.11.0’
micromamba activate FOSCAT
pip install foscat
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git

```

## Authors and acknowledgment

Authors: J.-M. Delouis, P. Campeti, T. Foulquier, J. Mangin, L. Mousset, T. Odaka, F. Paul, E. Allys

This work is part of the R & T Deepsee project supported by CNES. The authors acknowledge the heritage of the Planck-HFI consortium regarding data, software, knowledge. This work has been supported by the Programme National de Télédétection Spatiale (PNTS, http://programmes.insu.cnrs.fr/pnts/), grant n◦ PNTS-2020-08

## License

BSD 3-Clause License

Copyright (c) 2022, the Foscat developers All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

## Project status

It is a scientific driven development. We are open to any contributing development.
