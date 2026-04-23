import torch
import numpy as np
import argparse
import os

from fno import FourierNeuralOperator
from train import make_dataloaders, Evaluator
from visualization import plot_1d_realizations, plot_2d_realizations, plot_ensemble_statistics

def load_dataset(filename):
    return np.load(filename, allow_pickle=True).item()

def get_predictions(model, loader, device):
    model.eval()
    all_inputs = []
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            
            all_inputs.append(inputs)
            all_targets.append(targets)
            all_preds.append(preds)
            
    return torch.cat(all_inputs), torch.cat(all_targets), torch.cat(all_preds)

def validate(pde_type, model_path, data_path, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading test data from {data_path}...")
    test_data = load_dataset(data_path)
    
    if pde_type == "burgers":
        spatial_dim = 1
        nx = 64
        lx = 2 * np.pi
        x_grid = torch.linspace(0, lx, nx)
        
        # We need a dummy train/val dataset to make dataloaders because make_dataloaders 
        # expects train_data to compute stats. 
        # Alternatively, we can assume the test data has the same stats.
        # But to be exact, we should load train_data stats or just use test data for stats during pure validation.
        _, _, test_loader, _ = make_dataloaders(
            test_data, test_data, test_data, spatial_dim, x_grid, batch_size=batch_size
        )
        
        model = FourierNeuralOperator(
            in_channels=3, 
            out_channels=1, 
            hidden_width=32, 
            n_fourier_layers=4,
            n_modes_x=16
        ).to(device)
        
    elif pde_type == "ns":
        spatial_dim = 2
        nx, ny = 32, 32
        lx, ly = 2 * np.pi, 2 * np.pi
        x_grid = torch.linspace(0, lx, nx)
        y_grid = torch.linspace(0, ly, ny)
        
        _, _, test_loader, _ = make_dataloaders(
            test_data, test_data, test_data, spatial_dim, x_grid, y_grid=y_grid, batch_size=batch_size
        )
        
        model = FourierNeuralOperator(
            in_channels=4, 
            out_channels=1, 
            hidden_width=32, 
            n_fourier_layers=4,
            n_modes_x=12, 
            n_modes_y=12
        ).to(device)
    else:
        raise ValueError("Invalid PDE type. Choose 'burgers' or 'ns'.")

    print(f"Loading model weights from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    
    print("Running evaluation...")
    evaluator = Evaluator(model, spatial_dim=spatial_dim, device=device)
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)
    
    print("Generating visualizations...")
    inputs, targets, preds = get_predictions(model, test_loader, device)
    
    if spatial_dim == 1:
        plot_1d_realizations(inputs, targets, preds, x_grid.numpy(), num_samples=3, filename="val_burgers_realizations.png")
        plot_ensemble_statistics(targets, preds, spatial_dim, filename="val_burgers_ensemble_stats.png")
    else:
        plot_2d_realizations(inputs, targets, preds, num_samples=2, filename="val_ns_realizations.png")
        plot_ensemble_statistics(targets, preds, spatial_dim, filename="val_ns_ensemble_stats.png")
        
    print("Validation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate trained FNO models and generate visualizations.")
    parser.add_argument("--pde", type=str, choices=["burgers", "ns"], required=True, help="Target PDE (burgers or ns)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pt file)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test dataset (.npy file)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    validate(args.pde, args.model_path, args.data_path, args.batch_size)
