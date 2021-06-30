# CUDA Explicit MPM

## About the project

![Cover Image](image/MPMCUDA_Cover.png)

This project implements the explicit MPM algorithm with three 
material types, which are JELLO, SNOW and WATER in 3D. In addition, 
under the root directory you can also find Taichi (v0.7.22) 2D explicit
MPM implementation for these three material types. Full [Demo on Youtube]() and [Demo on Bilibili]().  

## Build this project

Currently, it's only tested and deployed on Ubuntu 20.04.1 LTS (Focal Fossa) with CUDA 11.1. In order to 
build this project, it is only necessary to run

```
python build_compile.py
```

If you have a hard time building this project, please feel free to submit an issue or contact me
through email (You can find me in my contact information on my Github account). I'll try my best
to help you with configuration.

## Use this project

The data files in .bego format would be generated under the 'animation' folder which is generated automatically 
by this application. They can be imported into other 3D software for rendering.

## Reference

* UPenn CIS563 2020 materials.
* **[GPU Optimization of Material Point Methods](https://github.com/kuiwuchn/GPUMPM)**
* **[MPM Simulation & Voxels Rendering](https://windqaq.github.io/MPM/)**


