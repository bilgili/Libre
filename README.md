[![Build Status](https://travis-ci.org/bilgili/Libre.svg?branch=master)](https://travis-ci.org/bilgili/Libre)

# Libre

![Libre](doc/images/livre_small.png)

Libre (Large-scale Interactive Volume Rendering Engine) is an out-of-core,
multi-node, multi-gpu volume rendering engine to visualise large
volumetric data sets. Libre is cloned from the Livre project in the Blue Brain
Project space [source code] (https://github.com/BlueBrain/Livre.git).

Libre be mainly used for developing new algorithms with more recent versions of
OpenGL, CUDA, Vulkan etc ( may be very unstable ) and is not meant to be alternative
to original Livre that is developed by the Blue Brain team.

It provides the following major features to facilitate rendering of large volumetric data sets:
* Plugin-based data sources
* Plugin-based renderers ( OpenGL 4.3 and CUDA 8.0 )
* Highly paralllel architecture for rendering, data uploading and computations based on
  pipeline architecture based on C++11 futures.
* Visualisation of pre-processed UVF format
  ([source code](https://github.com/SCIInstitute/Tuvok.git)) volume data sets.
* Multi-node, multi-gpu rendering (Currently only sort-first rendering)

To keep track of the changes between releases check the [changelog](doc/Changelog.md).

Contact: ahmetbilgili@gmail.com

## Known Bugs

Please file a [Bug Report](https://github.com/bilgili/Libre/issues) if you find new
issues which have not already been reported in
[Bug Report](https://github.com/bilgili/Libre/issues) page. If you find an already reported problem,
please update the corresponding issue with your inputs and outputs.

## About

The following platforms and build environments are tested:

* Linux: Ubuntu 16.04 (Makefile, x64) ( Tested with CUDA 8.0 and Nvidia GTX 970 )

