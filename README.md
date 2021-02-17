# ShNAPr
A collection of Python modules for developing **Sh**ell **N**onlinear **A**nalysis **Pr**ograms (pronounced "shnapper", similar to "snapper", a colloquial term for snapping turtles, which belong to a biological order best known for having _shells_.)  This library discretizes shell structures isogeometrically, and relies on [tIGAr](https://github.com/david-kamensky/tIGAr) and its associated dependencies, chiefly [FEniCS](https://fenicsproject.org/).  It is also recommended to use this library in conjunction with the advanced form compiler [TSFC](https://doi.org/10.1137/17M1130642), which may be installed for FEniCS by following the Singularity recipe provided in the tIGAr repository.  

This module was originally written to support the following paper, submitted to a special issue on open-source software for partial differential equations:
```
@article{Kamensky2021,
title = "Open-source immersogeometric analysis of fluid--structure interaction using {FEniCS} and {tIGAr}",
journal = "Computers \& Mathematics with Applications",
volume = "81",
pages = "634--648",
year = "2021",
note = "Development and Application of Open-source Software for Problems with Numerical PDEs",
issn = "0898-1221",
author = "D. Kamensky"
}
```