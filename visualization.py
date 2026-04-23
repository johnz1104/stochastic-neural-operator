import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def plot_1d_realizations(inputs, targets, preds, x_grid, num_samples=3, filename="burgers_realizations.png"):
    """Plot individual realizations for 1D problem"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(inputs))):
        u0 = inputs[i, 0].cpu().numpy()
        noise = inputs[i, 1].cpu().numpy()
        true_sol = targets[i, 0].cpu().numpy()
        pred_sol = preds[i, 0].cpu().numpy()
        
        ax = axes[i]
        ax.plot(x_grid, u0, 'k--', alpha=0.5, label='Initial Condition')
        ax.plot(x_grid, true_sol, 'b-', linewidth=2, label='True Solution')
        ax.plot(x_grid, pred_sol, 'r--', linewidth=2, label='FNO Prediction')
        
        ax.set_title(f'Realization {i+1}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved 1D realizations plot to {filename}")

def plot_2d_realizations(inputs, targets, preds, num_samples=2, filename="ns_realizations.png"):
    """Plot individual realizations for 2D problem"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(min(num_samples, len(inputs))):
        true_sol = targets[i, 0].cpu().numpy()
        pred_sol = preds[i, 0].cpu().numpy()
        error = np.abs(true_sol - pred_sol)
        
        # True
        im1 = axes[i][0].imshow(true_sol, cmap='RdBu_r', origin='lower')
        axes[i][0].set_title(f'True Vorticity (Sample {i+1})')
        plt.colorbar(im1, ax=axes[i][0], fraction=0.046, pad=0.04)
        
        # Predicted
        im2 = axes[i][1].imshow(pred_sol, cmap='RdBu_r', origin='lower')
        axes[i][1].set_title(f'Predicted Vorticity (Sample {i+1})')
        plt.colorbar(im2, ax=axes[i][1], fraction=0.046, pad=0.04)
        
        # Error
        im3 = axes[i][2].imshow(error, cmap='magma', origin='lower')
        axes[i][2].set_title(f'Absolute Error (Sample {i+1})')
        plt.colorbar(im3, ax=axes[i][2], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved 2D realizations plot to {filename}")

def plot_ensemble_statistics(targets, preds, spatial_dim, filename="ensemble_stats.png"):
    """Plot ensemble mean and variance comparisons"""
    if spatial_dim == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        true_mean = targets.mean(dim=0)[0].cpu().numpy()
        pred_mean = preds.mean(dim=0)[0].cpu().numpy()
        true_var = targets.var(dim=0)[0].cpu().numpy()
        pred_var = preds.var(dim=0)[0].cpu().numpy()
        
        ax1.plot(true_mean, 'b-', label='True Mean')
        ax1.plot(pred_mean, 'r--', label='Pred Mean')
        ax1.set_title('Ensemble Mean')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(true_var, 'b-', label='True Variance')
        ax2.plot(pred_var, 'r--', label='Pred Variance')
        ax2.set_title('Ensemble Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    elif spatial_dim == 2:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        true_mean = targets.mean(dim=0)[0].cpu().numpy()
        pred_mean = preds.mean(dim=0)[0].cpu().numpy()
        true_var = targets.var(dim=0)[0].cpu().numpy()
        pred_var = preds.var(dim=0)[0].cpu().numpy()
        
        # Mean
        im1 = axes[0, 0].imshow(true_mean, cmap='RdBu_r', origin='lower')
        axes[0, 0].set_title('True Mean')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(pred_mean, cmap='RdBu_r', origin='lower')
        axes[0, 1].set_title('Predicted Mean')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(np.abs(true_mean - pred_mean), cmap='magma', origin='lower')
        axes[0, 2].set_title('Mean Absolute Error')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Variance
        im4 = axes[1, 0].imshow(true_var, cmap='viridis', origin='lower')
        axes[1, 0].set_title('True Variance')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(pred_var, cmap='viridis', origin='lower')
        axes[1, 1].set_title('Predicted Variance')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(np.abs(true_var - pred_var), cmap='magma', origin='lower')
        axes[1, 2].set_title('Variance Absolute Error')
        plt.colorbar(im6, ax=axes[1, 2])
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved ensemble statistics plot to {filename}")

def plot_training_history(history, filename="training_history.png"):
    """Plot training and validation loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train']) + 1)
    
    ax1.plot(epochs, history['train'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val'], 'r-', label='Validation Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['lr'], 'k-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved training history plot to {filename}")
