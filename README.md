# Stable2DObject: A 2D Fluid Dynamics Playground

`Stable2DObject` is a Python-based fluid dynamics simulator designed to model fluid motion, diffusion, and interactions with obstacles in a 2D grid. It is particularly suited for educational, research, and visualization purposes. This implementation focuses on stability, efficiency, and diversity.

---

![Fluid Simulation Gif](gifs/wind_tunnel_dark.gif)
Applying no-slip boundary conditions, this GIF simulates the flow of dye around a rotating object while being subject to a constant force to the right. The velocity magnitude is plotted on the left, and the dye concentration on the right. 


## Features

- **2D Fluid Simulation**: Simulates the velocity field and dye dynamics on a 2D grid.
- **Diffusion and Advection**: Implements stable diffusion and semi-Lagrangian advection schemes.
- **Boundary Handling**: Supports no-slip boundary conditions and obstacle masking.
- **Sparse Matrix Solver**: Uses sparse matrix techniques for solving pressure correction equations.
- **Force Application**: Allows external force application to the velocity field.
- **Customizable Parameters**:
  - Grid resolution (ngrid)
  - Diffusion coefficient (diffusion)
  - Viscosity (viscosity)
  - Time step size (dt)
  - Dye diffusion rate (dye_diffusion)
  - External force (force)
---
## Requirements
To set up the environment, run `conda env create -f environment.yml`. Then, activate the environment with `conda activate stable_env`. Alternatively, you can install the dependencies by running `pip install -r requirements.txt`.

## Installation

Clone this repository:
```bash
git clone https://github.com/your-username/stable2dobject.git
cd stable2dobject
pip install -r requirements.txt
```


## ```diffuse.py```

For a sample run of the algorithm, run
```bash
python diffuse.py
```
and see ```gifs/fluid_test.gif``` for the output.


<h3 align="center">
  All boundary conditions applied.
</h3>

![Fluid Simulation Gif](gifs/wind_tunnel_dye_all.gif)

<h3 align="center">
  No boundary conditions applied.
</h3>

![Fluid Simulation Gif](gifs/wind_tunnel_dye_none.gif)


<h3 align="center">
  Dye under righward force.
</h3>

![Fluid Simulation Gif](gifs/fluid_with_dye_wind.gif)


<h3 align="center">
  Forceless Diffusion
</h3>

![Fluid Simulation Gif](gifs/fluid_with_dye_no_force.gif)
