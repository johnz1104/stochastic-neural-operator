# Stochastic Fourier Neural Operator

A Fourier Neural Operator for learning solution maps of stochastic partial differential equations (SPDEs). The model aims to learn to predict the solution field at a final time given an initial condition and a stochastic forcing realization for the stochastic Navier-Stokes equations. 

## Motivation

Stochastic PDEs model physical systems subject to random forcing — turbulent flows, noisy heat transport, uncertain material response. Traditional pseudo-spectral solvers produce accurate solutions but are computationally expensive: each new realization of the noise requires a full time integration. For tasks that require many realizations (uncertainty quantification, ensemble forecasting, optimization under uncertainty), this cost becomes prohibitive.

Neural operators offer a different tradeoff. Rather than solving the PDE from scratch each time, we train a model on solver-generated data to learn the map:

```
(initial condition, noise realization) -> solution at final time
```

Once trained, inference is a single forward pass, orders of magnitude faster than re-running the solver. The Fourier Neural Operator architecture is particularly well-suited here because its spectral convolution layers give every layer a global receptive field, matching the non-local nature of PDE solution operators.

## Problem Formulations

### 1D Stochastic Burgers Equation

```
du/dt + u * du/dx = nu * d^2u/dx^2 + sigma * dW/dt
```

A nonlinear convection-diffusion equation with additive stochastic forcing. Shares the core physics of 1D Navier-Stokes (nonlinear advection + viscous diffusion) while dropping pressure and incompressibility. Solved with a pseudo-spectral IMEX-Heun scheme: an integrating factor handles diffusion exactly, while Heun's method (a predictor-corrector) advances the nonlinear term.

### 2D Stochastic Navier-Stokes (Vorticity-Streamfunction)

```
d(omega)/dt + J(psi, omega) = nu * Lap(omega) + curl(f)
```

The full 2D incompressible Navier-Stokes equations reformulated via the curl. Working in vorticity-streamfunction variables eliminates pressure, automatically satisfies incompressibility, and reduces the system to a single scalar equation. Solved with a pseudo-spectral method using Crank-Nicolson for diffusion and Adams-Bashforth 2 for advection.

## Techniques

### Pseudo-Spectral Solvers (`solvers.py`)

- **Spectral spatial discretization**: fields are represented by their Fourier coefficients. Spatial derivatives become algebraic multiplications (`ik` for first derivative, `-k^2` for Laplacian), giving spectral accuracy.
- **2/3 dealiasing rule**: the quadratic nonlinearity (u^2 in Burgers, u * grad(omega) in NS) generates modes beyond the grid resolution. The 2/3 rule zeros high-frequency modes before computing products in physical space, preventing spurious energy transfer.
- **IMEX time integration**: diffusion (linear, stiff) is handled implicitly (integrating factor or Crank-Nicolson), while advection (nonlinear, non-stiff) is handled explicitly (Heun or Adams-Bashforth). This avoids the severe time-step restrictions of fully explicit schemes.
- **Spatially correlated noise**: stochastic forcing is generated in Fourier space with a Gaussian spectral amplitude profile controlled by a correlation length parameter, producing physically plausible colored noise fields.

### Fourier Neural Operator (`fno.py`)

- **Lifting layer**: pointwise (1x1 convolution) map from input channels to a high-dimensional feature space where the PDE mapping is easier to represent.
- **Spectral convolution layers**: learned complex weight matrices applied to the lowest Fourier modes. Each layer has a global receptive field — every spatial point can influence every other in a single layer. High-frequency modes above the cutoff are zeroed out.
- **Dual-path Fourier layers**: each layer combines a spectral convolution (global structure) with a pointwise convolution (local nonlinearities), summed and passed through GELU activation with a residual connection.
- **Projection layer**: maps features back to the physical output space.

### Training Pipeline (`train.py`)

- **Combined loss function**: weighted sum of relative L2 (per-realization accuracy), spectral loss (energy distribution across scales), and moment loss (ensemble mean/variance consistency).
- **AdamW optimizer** with cosine annealing learning rate schedule and linear warmup.
- **Gradient clipping** for training stability.
- **Evaluation metrics**: per-realization relative L2 error (mean, median, 95th percentile) and ensemble statistics (mean field error, variance field error).

## Project Structure

```
fno.py      - FNO architecture (spectral convolutions, Fourier layers, full model)
solvers.py  - pseudo-spectral PDE solvers, noise generation, dataset generation
train.py    - dataset/dataloader, loss functions, trainer, evaluator
main.py     - CLI orchestrator for data generation and model training

## Usage

You can use `main.py` to generate data and train the model for either the 1D Burgers or 2D Navier-Stokes equations.

### Data Generation

Generate a dataset of SPDE solution pairs (saved as `.npy` files):

```bash
# Generate 1000 samples for the 1D Burgers equation
python main.py --pde burgers --mode generate --num_samples 1000

# Generate 500 samples for the 2D Navier-Stokes equation
python main.py --pde ns --mode generate --num_samples 500
```

### Model Training

Train the Fourier Neural Operator on the generated datasets:

```bash
# Train on the Burgers dataset for 100 epochs
python main.py --pde burgers --mode train --epochs 100 --batch_size 32

# Train on the Navier-Stokes dataset for 200 epochs
python main.py --pde ns --mode train --epochs 200 --batch_size 16
```
```