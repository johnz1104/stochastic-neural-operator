import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# DATASET
class SPDEDataset(Dataset):
    """
    pytorch dataset for SPDE neural operator training
        - takes raw PDE data, normalizes it, and returns training samples
        (one training sample is one example that the nmodel learns from)

    each sample is a typle: (input_tensor, output_tensor)
        - input_tensor has channels [initial conditions, noise, coordiantes]
        - output_tensor is the solution field
    """

    def __init__(self, data:dict, spatial_dim: int, x_grid, y_grid = None, stats: dict = None):
        """
        Args:
            data: dict with 'u0', 'noise', 'solutions' from DataGenerator
            spatial_dim: 1 or 2
            x_grid: (Nx,) spatial coordinates
            y_grid: (Ny,) spatial coordinates (2D only)
            stats: normalization stats from training set (None = compute from this data)
        """
        # store raw data
        self.u0 = data['u0']
        self.noise = data['noise']
        self.solutions = data['solutions']

        # store geometry
        self.spatial_dim = spatial_dim
        self.x_grid = x_grid
        self.y_grid = y_grid
 
        # compute or apply normalization
        if stats is None:
            self.stats = {
                'u0_mean': self.u0.mean(), 'u0_std': self.u0.std() + 1e-8,
                'noise_mean': self.noise.mean(), 'noise_std': self.noise.std() + 1e-8,
                'sol_mean': self.solutions.mean(), 'sol_std': self.solutions.std() + 1e-8,
            }
        else:
            self.stats = stats
 
        # normalize
        self.u0 = (self.u0 - self.stats['u0_mean']) / self.stats['u0_std']
        self.noise = (self.noise - self.stats['noise_mean']) / self.stats['noise_std']
        self.solutions = (self.solutions - self.stats['sol_mean']) / self.stats['sol_std']

    def __len__(self):
        """ returns number of samples """
        return self.u0.shape[0]
    
    def _get_1d(self, index):
        """
        Construct a single 1D training sample.

        Each sample consists of stacked input channels and the corresponding
        target solution:
        Channels:
            0: normalized initial condition u0(x)
            1: normalized noise field ξ(x)
            2: normalized spatial coordinate x ∈ [0, 1]

        Args:
            index (int):
                Index of the sample in the dataset.

        Returns:
            input_tensor (torch.Tensor):
                Shape (3, Nx). Channel-stacked input for the model.
            output_tensor (torch.Tensor):
                Shape (1, Nx). Normalized target solution at final time.
        """
        batch_u0 = self.u0[index].unsqueeze(0)
        batch_noise = self.noise[index].unsqueeze(0) 
        x_norm = (self.x_grid / self.x_grid.max()).unsqueeze(0).float()

        input_tensor = torch.cat([batch_u0, batch_noise, x_norm], dim=0)
        output_tensor = self.solutions[index].unsqueeze(0)
        return input_tensor.float(), output_tensor.float()

    def _get_2d(self, idx):
        """
        Construct a single 2D training sample.

        Each sample consists of stacked input channels and the corresponding
        target solution:
            Channels:
                0: normalized initial condition u0(x, y)
                1: normalized noise field ξ(x, y)
                2: normalized x-coordinate grid ∈ [0, 1]
                3: normalized y-coordinate grid ∈ [0, 1]

        Args:
            idx (int):
                Index of the sample in the dataset.

        Returns:
            input_tensor (torch.Tensor):
                Shape (4, Nx, Ny). Channel-stacked input for the model.

            output_tensor (torch.Tensor):
                Shape (1, Nx, Ny). Normalized target solution at final time.
        """
        batch_u0 = self.u0[idx].unsqueeze(0)
        batch_noise = self.noise[idx].unsqueeze(0)
 
        x_norm = self.x_grid / self.x_grid.max()
        y_norm = self.y_grid / self.y_grid.max()
        xx, yy = torch.meshgrid(x_norm, y_norm, indexing='ij')
        xx = xx.unsqueeze(0).float()
        yy = yy.unsqueeze(0).float()
 
        input_tensor = torch.cat([batch_u0, batch_noise, xx, yy], dim=0)
        output_tensor = self.solutions[idx].unsqueeze(0)
        return input_tensor.float(), output_tensor.float()
    
    def __getitem__(self, index):
        """ returns one training sample """
        if self.spatial_dim == 1:
            return self._get_1d(index)
        else:
            return self._get_2d(index)

def make_dataloaders(train_data, val_data, test_data, spatial_dim, x_grid, y_grid = None, batch_size = 32):
    """
    creates train/val/test Pytorch DataLoaders from raw datasets with proper normalization
        - allows learning model to pull batches of input/target automatically during training
    """

    train_ds = SPDEDataset(train_data, spatial_dim, x_grid, y_grid)
    val_ds = SPDEDataset(val_data, spatial_dim, x_grid, y_grid, stats=train_ds.stats)
    test_ds = SPDEDataset(test_data, spatial_dim, x_grid, y_grid, stats=train_ds.stats)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
 
    return train_loader, val_loader, test_loader, train_ds.stats


# LOSS FUNCTIONS
class RelativeL2Loss:
    """
    relative L2 error (nomralized mean-square error (MSE))

    L = mean_i( ||pred_i - true_i||_2 / ||true_i||_2 )
 
    using relative error prevents the loss from being dominated by high-amplitude samples
    """
    def __call__(self, pred, target):
        batch = pred.shape[0]
        pred_flat = pred.reshape(batch, -1)
        target_flat = target.reshape(batch, -1)
        diff_norm = torch.norm(pred_flat - target_flat, dim=1)
        target_norm = torch.norm(target_flat, dim=1) + 1e-8
        return (diff_norm / target_norm).mean()

class SpectralLoss:
    """
    loss on the energy spectrum of predicted vs true fields

    compares the distribution of energy across spatial scales using log-space MSE
        - this is very important for turbulence
    """

    def __init__(self, spatial_dim: int):
        self.spatial_dim = spatial_dim
 
    def __call__(self, pred, target):
        if self.spatial_dim == 1:
            spec_pred = torch.abs(torch.fft.rfft(pred, dim=-1))**2
            spec_true = torch.abs(torch.fft.rfft(target, dim=-1))**2
        else:
            spec_pred = torch.abs(torch.fft.rfft2(pred, dim=(-2, -1)))**2
            spec_true = torch.abs(torch.fft.rfft2(target, dim=(-2, -1)))**2
    
        #average over batch and channels
        spec_pred = spec_pred.mean(dim=(0, 1))
        spec_true = spec_true.mean(dim=(0, 1))
 
        #log-space MSE (better for spectra spanning orders of magnitude)
        return nn.functional.mse_loss(torch.log1p(spec_pred), torch.log1p(spec_true))
    
class MomentLoss: 
    """
    statistical consistency loss across a mini-batch

    compares the batch mean and variance of prediction vs truth
        - encourages model to reproduce ensemble statistics and not just individual realizations
    only useful with large batch_size
    """
    def __call__(self, pred, target):
        mean_loss = nn.functional.mse_loss(pred.mean(dim=0), target.mean(dim=0))
        var_loss = nn.functional.mse_loss(pred.var(dim=0), target.var(dim=0))
        return mean_loss + var_loss

class CombinedLoss:
    """
    weighted sum of all loss components

    total = w_pw * pointwise + w_sp * spectral + w_mom * moment

    the weights control what the model prioritizes during training:
        - high w_pw: focus on per-realization accuracy
        - high w_sp: focus on getting the energy spectrum right
        - high w_mom: focus on ensemble statistics (mean, variance fields)
    """

    def __init__(self, spatial_dim: int, w_pointwise=1.0, w_spectral=0.1, w_moment=0.05):
        self.pointwise = RelativeL2Loss()
        self.spectral = SpectralLoss(spatial_dim)
        self.moment = MomentLoss()
        self.w_pw = w_pointwise
        self.w_sp = w_spectral
        self.w_mom = w_moment
 
    def __call__(self, pred, target):
        loss_pw = self.pointwise(pred, target)
        loss_sp = self.spectral(pred, target)
 
        total = self.w_pw * loss_pw + self.w_sp * loss_sp
 
        #only compute moment loss if batch is large enough
        if pred.shape[0] >= 8:
            loss_mom = self.moment(pred, target)
            total = total + self.w_mom * loss_mom
        else:
            loss_mom = torch.tensor(0.0)
 
        return total, {
            'total': total.item(),
            'pointwise': loss_pw.item(),
            'spectral': loss_sp.item(),
            'moment': loss_mom.item(),
        }
    
# TRAINER 
class Trainer: 
    """
    training loop for the neural operator
    
    the training process (what happens each iteration):
        1. sample a mini-batch of (input, target) pairs
        2. forward pass: pred = model(input)
        3. compute loss: how wrong is the prediction?
        4. backward pass: compute gradients d(loss)/d(parameters) via backpropagation
        5. optimizer step: update parameters using gradient descent (adam variant)
        6. repeat until loss converges
    """

    def __init__(self, model, spatial_dim: int, 
                learning_rate: float = 1e-3, 
                weight_decay: float = 1e-4,
                n_epochs: int = 500,
                warmup_epochs: int = 5, 
                grad_clip: float = 1.0,
                device: str = 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        
        # loss function (weighted combinatino of pointwise + spectral + moment)
        self.criterion = CombinedLoss(spatial_dim)

        # Adam = adaptive moemnt estimation 
        # AdamW = Adam with weight decay

        # AdamW optimizer:
        # adam maintains per-parameter adaptive learning rates using running
        # averages of the gradient (momentum) and squared gradient (scaling)
        # the 'W' variant decouples weight decay from the adaptive learning rate

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay, 
            betas=(0.9, 0.999)      #momentum terms
        )

        #learning rate scheduler: cosine annealing
        #starts at lr, decays following a cosine curve to eta_min
        #this gives aggressive exploration early, fine-tuning late
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6
        )
 
        #tracking
        self.best_val_loss = float('inf')
        self.history = {'train': [], 'val': [], 'lr': []}

    def _warmup_lr(self, epoch, step, steps_per_epoch):
        """ 
        linear learning rate warmup for the first few epochs
        """

        if epoch < self.warmup_epochs:
            total = self.warmup_epochs * steps_per_epoch
            current = epoch * steps_per_epoch + step
            base_lr = self.optimizer.defaults['lr']
            lr = base_lr * (current + 1) / total 

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, train_loader):
        """ run one training epoch """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self._warmup_lr(self.current_epoch, step, len(train_loader))

            # forward pass
            pred = self.model(inputs)
            loss, loss_dict = self.criterion(pred, targets)

            # backward pass: compute d(loss)/d(theta) for all parameters
            # this is backpropagation 
            #   - chain rule applied through the computation graph, from loss back through every layer
            self.optimizer.zero_grad()      # clear old gradients
            loss.backward()                 # compute new gradients

            # gradient clipping: cap gradient magnitude to prevent instability
            # if ||grad|| > clip_value, scale it down to clip_value
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Adam optimizer step: (gradient descent)
            # this computes: theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
            #   where -    
            #       m_hat ~ running average of gradient (direction)      
            #       v_hat ~ running average of gradient^2 (scale)
            self.optimizer.step()
 
            total_loss += loss_dict['total']
            n_batches += 1
 
        return total_loss / n_batches

    def val_epoch(self, val_loader): 
        """run one validation epoch (no gradient computation)"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # torch.no_grad() tells pytorch not to track gradients
        # saves memory and computation during evaluation
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
 
                pred = self.model(inputs)
                loss, loss_dict = self.criterion(pred, targets)
 
                total_loss += loss_dict['total']
                n_batches += 1
 
        return total_loss / max(n_batches, 1)

    def fit(self, train_loader, val_loader, log_interval: int = 20):
        """
        full training loop (where the model actually learns)
        
        each epoch: 
            1. iterate through all training batches (gradient descent steps)
            2. evaluate on validation set (no learning, just measuring)
            3. update learning rate schedule
            4. save th mode if its the best so far
        """
        print(f"training on {self.device} for {self.n_epochs} epochs")
        print(f"  train batches: {len(train_loader)}, val batches: {len(val_loader)}")
        
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)

            #step the learning rate scheduler (after warmup period)
            lr = self.optimizer.param_groups[0]['lr']
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # record history 
            self.history['train'].append(train_loss)
            self.history['val'].append(val_loss)
            self.history['lr'].append(lr)
 
            #save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
 
            #logging
            if (epoch + 1) % log_interval == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1:4d}/{self.n_epochs} | "
                      f"train: {train_loss:.6f} | val: {val_loss:.6f} | "
                      f"lr: {lr:.2e} | {elapsed:.1f}s")
 
        self.save_checkpoint('final_model.pt')
        print(f"done. best val loss: {self.best_val_loss:.6f}")
        return self.history
    
    def save_checkpoint(self, filename: str):
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }, os.path.join('checkpoints', filename))


    def load_checkpoint(self, filename: str):
        ckpt = torch.load(os.path.join('checkpoints', filename), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        print(f"loaded checkpoint: {filename}")
    
# EVALUATION
class Evaluator: 
    """
    evaluation metrics for a trainsed neural operator

    computes both per-realization accuracy and ensemble statistics 
        - assesses whether model captures the statistical strcuture of SPDE solution process
    """
    def __init__(self, model, spatial_dim: int, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.spatial_dim = spatial_dim
        self.device = device
    
    def evaluate(self, test_loader):
        """
        run full evaluation on test set

        per-relization: relative L2 error for each sample
        eensemble: mean field error, variance field error, spectrum error
        """
        all_preds = []
        all_targets = []
        errors = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
 
                pred = self.model(inputs)
 
                all_preds.append(pred.cpu())
                all_targets.append(targets.cpu())
 
                # per-sample relative L2 error
                batch = pred.shape[0]
                p_flat = pred.reshape(batch, -1)
                t_flat = targets.reshape(batch, -1)
                err = (torch.norm(p_flat - t_flat, dim=1) / (torch.norm(t_flat, dim=1) + 1e-8))
                errors.append(err)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        errors = torch.cat(errors)

        # ensemble statistics
        pred_mean = all_preds.mean(dim=0)
        true_mean = all_targets.mean(dim=0)
        mean_err = (torch.norm(pred_mean - true_mean)
                    / (torch.norm(true_mean) + 1e-8)).item()
 
        pred_var = all_preds.var(dim=0)
        true_var = all_targets.var(dim=0)
        var_err = (torch.norm(pred_var - true_var)
                   / (torch.norm(true_var) + 1e-8)).item()
 
        results = {
            'mean_rel_l2': errors.mean().item(),
            'median_rel_l2': errors.median().item(),
            'p95_rel_l2': errors.quantile(0.95).item(),
            'max_rel_l2': errors.max().item(),
            'mean_field_error': mean_err,
            'variance_field_error': var_err,
        }
 
        return results


    def print_results(self, results: dict):
        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  mean relative L2:      {results['mean_rel_l2']:.6f}")
        print(f"  median relative L2:    {results['median_rel_l2']:.6f}")
        print(f"  95th percentile L2:    {results['p95_rel_l2']:.6f}")
        print(f"  mean field error:      {results['mean_field_error']:.6f}")
        print(f"  variance field error:  {results['variance_field_error']:.6f}")
        print("=" * 55)