
# SNPIT Specutils

This is a compendium of utility functions to use the SNPIT WFSS pipeline for Roman/WFI. 

## Contents
1. optical model: this describes the spectral trace and dispersion.
2. background: functions to subtract a `global sky image` from a dispersed image.

## Caveats
For the `optical model`, you may get this error message:

> OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

This is safe to ignore and will be addressed by `numba` updates.
