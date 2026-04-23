import numpy as np
import torch
import argparse
import os

from solvers import StochasticBurgers1D, StochasticNavierStokes2D, StochasticNoise, DataGenerator
from fno import FourierNeuralOperator
from train import make_dataloaders, Trainer, Evaluator

def save_dataset(data, filename):
    np.save(filename, data)
    print(f"Dataset saved to {filename}")

def load_dataset(filename):
    return np.load(filename, allow_pickle=True).item()

def run_1d_burgers(args):
    """
    Control flow for 1D Stochastic Burgers Equation
    """
    nx = 64 # Grid points
    lx = 2 * np.pi
    
    if args.mode == "generate":
        print("Initializing 1D Burgers Solver...")
        solver = StochasticBurgers1D(lx=lx, nx=nx, t=0.5, nt=200, nu=0.01)
        noise = StochasticNoise(sigma=0.05, correlation_length=0.3, seed=42)
        generator = DataGenerator(solver, noise)
        
        print(f"Generating {args.num_samples} samples...")
        n_train = int(args.num_samples * 0.8)
        n_val = int(args.num_samples * 0.1)
        n_test = args.num_samples - n_train - n_val
        
        train_data, val_data, test_data = generator.generate_splits(n_train, n_val, n_test)
        
        save_dataset(train_data, 'burgers_train.npy')
        save_dataset(val_data, 'burgers_val.npy')
        save_dataset(test_data, 'burgers_test.npy')
        
    elif args.mode == "train":
        print("Loading datasets...")
        train_data = load_dataset('burgers_train.npy')
        val_data = load_dataset('burgers_val.npy')
        test_data = load_dataset('burgers_test.npy')
        
        x_grid = torch.linspace(0, lx, nx)
        spatial_dim = 1
        
        train_loader, val_loader, test_loader, stats = make_dataloaders(
            train_data, val_data, test_data, spatial_dim, x_grid, batch_size=args.batch_size
        )
        
        in_channels = 3 # u0, noise, x_norm
        out_channels = 1
        
        print("Initializing Fourier Neural Operator...")
        model = FourierNeuralOperator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            hidden_width=32, 
            n_fourier_layers=4,
            n_modes_x=16
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Initializing Trainer...")
        trainer = Trainer(model, spatial_dim=spatial_dim, device=device, n_epochs=args.epochs)
        
        print("Starting training...")
        history = trainer.fit(train_loader, val_loader)
        
        print("Evaluating...")
        evaluator = Evaluator(model, spatial_dim=spatial_dim, device=device)
        evaluator.evaluate(test_loader)
        
        trainer.save_checkpoint("burgers_final.pt")

def run_2d_navier_stokes(args):
    """
    Control flow for 2D Stochastic Navier-Stokes Equation
    """
    nx, ny = 32, 32
    lx, ly = 2 * np.pi, 2 * np.pi
    
    if args.mode == "generate":
        print("Initializing 2D Navier-Stokes Solver...")
        solver = StochasticNavierStokes2D(lx=lx, ly=ly, nx=nx, ny=ny, t=1.0, nt=100, nu=1e-3)
        noise = StochasticNoise(sigma=0.1, correlation_length=0.5, seed=42)
        generator = DataGenerator(solver, noise)
        
        n_train = int(args.num_samples * 0.8)
        n_val = int(args.num_samples * 0.1)
        n_test = args.num_samples - n_train - n_val
        
        print(f"Generating {args.num_samples} samples...")
        train_data, val_data, test_data = generator.generate_splits(n_train, n_val, n_test)
        
        save_dataset(train_data, 'ns_train.npy')
        save_dataset(val_data, 'ns_val.npy')
        save_dataset(test_data, 'ns_test.npy')
        
    elif args.mode == "train":
        print("Loading datasets...")
        train_data = load_dataset('ns_train.npy')
        val_data = load_dataset('ns_val.npy')
        test_data = load_dataset('ns_test.npy')
        
        x_grid = torch.linspace(0, lx, nx)
        y_grid = torch.linspace(0, ly, ny)
        spatial_dim = 2
        
        train_loader, val_loader, test_loader, stats = make_dataloaders(
            train_data, val_data, test_data, spatial_dim, x_grid, y_grid=y_grid, batch_size=args.batch_size
        )
        
        in_channels = 4 # u0, noise, x_norm, y_norm
        out_channels = 1
        
        print("Initializing Fourier Neural Operator...")
        model = FourierNeuralOperator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            hidden_width=32, 
            n_fourier_layers=4,
            n_modes_x=12, 
            n_modes_y=12
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Initializing Trainer...")
        trainer = Trainer(model, spatial_dim=spatial_dim, device=device, n_epochs=args.epochs)
        
        print("Starting training...")
        history = trainer.fit(train_loader, val_loader)
        
        print("Evaluating...")
        evaluator = Evaluator(model, spatial_dim=spatial_dim, device=device)
        evaluator.evaluate(test_loader)
        
        trainer.save_checkpoint("ns_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fourier Neural Operator for Stochastic PDEs")
    parser.add_argument("--pde", type=str, choices=["burgers", "ns"], required=True, help="Target PDE (burgers or ns)")
    parser.add_argument("--mode", type=str, choices=["generate", "train"], required=True, help="Execution mode (generate or train)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    
    if args.pde == "burgers":
        run_1d_burgers(args)
    elif args.pde == "ns":
        run_2d_navier_stokes(args)
