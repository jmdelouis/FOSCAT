# foscat

A python package dedicated to image component separation based on scattering transform analysis designed for high performance computing.

## the concept

The foscat genesis has been built to synthesise data (2D or Healpix) using Cross Scattering Transform. For a detailed method description please refer to https://arxiv.org/abs/2207.12527. This algorithm could be effectively usable for component separation (e.g. denoising).

A demo package for this process can be found at https://github.com/jmdelouis/FOSCAT_DEMO.


## usage


To generate test files run the following lines in python:

```pycon
>>> import foscat.build_demo as dem

>>> dem.genDemo()

>>> quit()
```

run 2D test

```sh
python test2D.py
```

to plot results

```sh
python test2Dplot.py
```

run Healpix test

```sh
python testHealpix.py
```

to plot results

```sh
python testHplot.py
```

Note: If mpi is available you can run testHealpix_mpi.py that uses 3 nodes to do the same computation than tesHealpix.py

```sh
mpirun -np 3 testHealpix_mpi.py
```

## Authors and acknowledgment

Authors: J.-M. Delouis, T. Foulquier, L. Mousset, T. Odaka, F. Paul, E. Allys

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
