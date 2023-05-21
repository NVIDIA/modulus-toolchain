[PINN Examples](../index.md)

# Wave Equation Modeling and Inversion using PINNs
Contributors: Guillaume Barnier (NVIDIA) and Pavel Dimitrov (NVIDIA)

---

## Description
The following examples show how to use Modulus Tool Chain (MTC) and Physics-Informed Neural Nets (PINN) to solve wave-equation (WE) partial differential equations (PDE) in 1D/2D. We tackle two different types of problems:

1. **Forward modeling**: given some initial/boundary conditions and a seismic velocity model, we train a neural net to predict the pressure field at any time/position in the domain of interest. 

2. **Velocity inversion**: given some initial/boundary conditions and some **recorded pressure data**, we train a neural net to predict 
    - The pressure field at any time/position, and 
    - The **seismic velocity model**

## 1D Wave Equation 
### 1. [Forward Modeling](we-1d-fwd/we-1d-fwd.md)
This collection of 1D examples is a translation of the 1D wave equation problem in the original [Modulus documentation](https://docs.nvidia.com/deeplearning/modulus/user_guide/foundational/1d_wave_equation.html). We provide four applications with different scenarios:  
- Constant velocity
- Non-constant velocity
- Constant velocity and velocity is an input to the NN
- Non-constant velocity parametrized by one parameter, which is an input to the NN

### 2. [Velocity Inversion](we-1d-inv/we-1d-inv.md)
- Constant velocity
- 2-layer model
- 3-layer model
- 4-layer model

## 2D Wave Equation 
### 1. [Forward Modeling](we-2d-fwd/we-2d-fwd.md)
### 2. Velocity Inversion

