# Stable2DObject: A 2D Fluid Dynamics Playground

`Stable2DObject` is a Python-based fluid dynamics simulator designed to model fluid motion, diffusion, and interactions with obstacles in a 2D grid. It is particularly suited for educational, research, and visualization purposes. This implementation focuses on stability, efficiency, and modularity.

---

## Features

- **Customizable Grid Resolution**: Choose the level of detail with adjustable grid sizes.
- **Physics Simulation**:
  - Velocity advection, diffusion, and projection.
  - Support for obstacles and no-slip boundary conditions.
  - Dye diffusion for visualizing flow.
- **Numerical Stability**: Uses semi-Lagrangian advection and sparse matrix solvers to ensure accuracy and performance.
- **Modular Design**: Easily extensible for additional physics or custom features.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/stable2dobject.git
   cd stable2dobject
