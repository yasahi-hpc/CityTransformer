# CityTransformer

_CityTransformer_ is designed to predict the plume concentrations in the urban area under uniform flow condition.
It has two distinct input layers: Transformer layers for time series data and convolutional layers for image-like data.
The inputs of the network are realistically available data such as the the building shapes and source locations and time series monitoring data a few observation stations. 

# Usage

## Installation
This code relies on the following packages. As a deeplearing framework, we use [PyTorch](https://pytorch.org).
- Install Python libraries
[numpy](https://numpy.org), [PyTorch](https://pytorch.org), [xarray](http://xarray.pydata.org/en/stable/), [horovod](https://github.com/horovod/horovod) and [netcdf4](https://github.com/Unidata/netcdf4-python)

- Clone this repo  
```git clone https://github.com/yasahi-hpc/CityTransformer.git```

## Prepare dataset

## Training
For training, it is recommended to use multiple Nvidia GPUs (12 GB memory or larger). 
We have trained the moel on [Nvidia V100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) GPUs.  
For the CNN architecture, we have prepared `Res-Net` and `U-Net` architectures. `Transformer` and `MLP` layers are available to encode time series data. 
The model can be set by the command line arguments.


## Inference

# Citations
```bibtex
@article{OnoderaBLM2021,
    title = {{Real-time tracer dispersion simulation in Oklahoma City using locally-mesh refined lattice Boltzmann method}},
    year = {2021},
    journal = {Boundary-Layer Meteorology},
    author = {Onodera, Naoyuki and Idomura, Yasuhiro and Hasegawa, Yuta and Nakayama, Hiromasa and Shimokawabe, Takashi and Aoki, Takayuki},
    publisher = {Springer Netherlands},
    doi = {10.1007/s10546-020-00594-x},
    issn = {1573-1472},
    keywords = {Adaptive mesh refinement, Large-eddy simulation, Lattice Boltzmann method}
}
```

```bibtex
@article{OnoderaMEJ2020,
  title={Locally mesh-refined lattice Boltzmann method for fuel debris air cooling analysis on GPU supercomputer},
  author={Naoyuki ONODERA and Yasuhiro IDOMURA and Shinichiro UESAWA and Susumu YAMASHITA and Hiroyuki YOSHIDA},
  journal={Mechanical Engineering Journal},
  volume={7},
  number={3},
  pages={19-00531-19-00531},
  year={2020},
  doi={10.1299/mej.19-00531}
}
```

```bibtex
@article{OnoderaICONE2018,
    author = {Onodera, Naoyuki and Idomura, Yasuhiro},
    title = "{Acceleration of Plume Dispersion Simulation Using Locally Mesh-Refined Lattice Boltzmann Method}",
    volume = {Volume 8: Computational Fluid Dynamics (CFD); Nuclear Education and Public Acceptance},
    series = {International Conference on Nuclear Engineering},
    year = {2018},
    month = {07},
    doi = {10.1115/ICONE26-82145},
    url = {https://doi.org/10.1115/ICONE26-82145},
    note = {V008T09A034},
    eprint = {https://asmedigitalcollection.asme.org/ICONE/proceedings-pdf/ICONE26/51524/V008T09A034/2457794/v008t09a034-icone26-82145.pdf},
}
```

```bibtex
@article{OnoderaScalA2018,
    title = {{Communication Reduced Multi-time-step Algorithm for Real-time Wind Simulation on GPU-based Supercomputers}},
    year = {2018},
    journal = {2018 IEEE/ACM 9th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (ScalA)},
    author = {Onodera, Naoyuki and Idomura, Yasuhiro and Ali, Yussuf and Shimokawabe, Takashi},
    pages = {9--16},
    isbn = {9781728101767}
}
```
