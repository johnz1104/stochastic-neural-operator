import torch 
import torch.nn as nn
import numpy as np


def build_input_tensor1D(u0, noise_spatial, x_grid: torch.Tensor):
    """
    assemble input tensor for 1D problems by stacking channels

    channels: 
        0: intiail condition u0(x)
        1: time-averaged noise field xi(x)
        2: spatial coordinate x (normalized to [0,1])

    Args:
        u0: (batch, nx) initial conditions
        noise_spatial: (batch, nx) time-averaged noise
        x_grid: (nx,) spatial coordinate grid
 
    Returns:
        (batch, 3, nx) input tensor
    """

    batch = u0.shape[0] 

    # normalize coordinates to [0,1]
    x_norm = (x_grid / x_grid.max()).unsqueeze(0).expand(batch, -1)     # (batch, Nx)

    # stack: (batch, 3, nx)
    return torch.stack([u0, noise_spatial, x_norm], dim = 1)

def build_input_tensor2D(omega0, noise_spatial, x_grid, y_grid):
    """
    assemble the input tensor for 2D problems
 
    channels:
        0: initial vorticity omega0(x, y)
        1: time-averaged noise field xi(x, y)
        2: x-coordinate grid (normalized)
        3: y-coordinate grid (normalized)
 
    Args:
        omega0: (batch, nx, ny) initial vorticity
        noise_spatial: (batch, nx, ny) time-averaged noise
        x_grid: (nx,) x coordinates
        y_grid: (ny,) y coordinates
 
    Returns:
        (batch, 4, nx, ny) input tensor
    """
    batch = omega0.shape[0]
    nx, ny = omega0.shape[1], omega0.shape[2]
 
    # coordinate grids normalized to [0, 1]
    x_norm = x_grid / x_grid.max()
    y_norm = y_grid / y_grid.max()
    xx, yy = torch.meshgrid(x_norm, y_norm, indexing='ij')      # (nx, ny) each
    xx = xx.unsqueeze(0).expand(batch, -1, -1)                  # (batch, nx, ny)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)
 
    # stack: (batch, 4, nx, ny)
    return torch.stack([omega0, noise_spatial, xx, yy], dim=1)



class SpectralConv1d(nn.Module):
    """
    learned linear transform applied in 1D fourier space

    SpectralConv1d: just one FNO layer (the spectral part)

    key component to FNO:
        - learn the weight matrix that operates on fourier coefficints directly
        - this gives the model global receptive field in a single layer

    the forward pass: 
        1. FFT input features 
        2. multiply lowest n_modes fourier coefficients by learned complex weights 
        3. inverse FFT back to physical space 
        4. high-freq modes (above n_modes) are zeroed out
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_channels = in_channels          # number of input fields 
        self.out_channels = out_channels        # number of output fields 
        self.n_modes = n_modes                  # how many Fourier modes you keep

        # operator W(k)
        # learned complex weights (in_c, out_c, n_modes, 2) where 2 means [re, im]
        scale = 1.0/ (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes, 2))

    def forward(self, x):
        """
        spectral layer: 
            - takes signal and turns into Fourier modes
            - learns how to modify low-frequency modes
            - throws away the rest (negative frequencies) 
            - turns result back to physical space
        
        input:
        x: (batch, in_channels, nx)

        returns: (batch, out_channels, nx)
        """
        batch, _, nx = x.shape

        # to fourier space (only keeps positive frequencies)
        # this results in fourier dimension becoming nx // 2 + 1
        x_ft = torch.fft.rfft(x, dim=1)

        # allocate output in fourier space
        out_ft = torch.zeros(batch, self.out_channels, nx // 2 + 1, dtype=torch.cfloat, device = x.device)

        # multiply low-freq by leanred complex weights
        #einsum: batch(b), in_channel(i), mode(m) x in_channel(i), out_channel(o), mode(m)
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, :self.n_modes] = torch.einsum('bim,iom->bom', x_ft[:, :, :self.n_modes], w)
 
        #back to physical space
        return torch.fft.irfft(out_ft, n = nx, dim=-1)
    

class SpectralConv2d(nn.Module):
    """
    learned linear transform applied in 2D fourier space

    same as 1D, except 2D real FFT has different structure:
        - kx runs over all modes 
        - ky only covers positive frequencies (since input is real, negative ky modes are comjugate symmetric)
    
    we learn two weight tensors for "corners" of 2D FFT output: 
        - weight1: low kx, low ky (positive kx modes)
        - weight2: high kx (negative kx modes), low ky
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes_x: int, n_modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
 
        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, 2))
        self.weight2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, 2))
    
    def forward(self, x):
        """
        x: (batch, in_channels, nx, ny)
        returns: (batch, out_channels, nx, ny)
        """
        batch, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
 
        out_ft = torch.zeros(batch, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
 
        w1 = torch.view_as_complex(self.weight1)
        w2 = torch.view_as_complex(self.weight2)
 
        #corner 1: low kx, low ky
        out_ft[:, :, :self.n_modes_x, :self.n_modes_y] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :self.n_modes_x, :self.n_modes_y], w1)
        
        #corner 2: high (negative) kx, low ky
        out_ft[:, :, -self.n_modes_x:, :self.n_modes_y] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, -self.n_modes_x:, :self.n_modes_y], w2)
 
        return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))

class FourierLayer(nn.Module):
    """
    single layer of the FNO: spectral convolution + local convolution + activation

    the fourier layer combines two paths:
        global path: spectral convolution (operates on fourier modes -> captures long-range)
        local path: 1x1 convolution (pointwise -> captures local nonlinearities)
 
    output = activation(spectral_conv(x) + local_conv(x)) + x
        - where x is the residual connection
 
    the residual connection (adding x back) makes training deeper networks stable
        - the network learns corrections to the identity operator
    """
    
    def __init__(self, width: int, n_modes_x: int, n_modes_y: int = None):
        """
        Args:
            width: number of feature channels
            n_modes_x: fourier modes to keep in x
            n_modes_y: fourier modes in y (None for 1D)
        """
        super().__init__()
        self.is_2d = n_modes_y is not None
 
        if self.is_2d:
            self.spectral = SpectralConv2d(width, width, n_modes_x, n_modes_y)
            self.local = nn.Conv2d(width, width, kernel_size=1)
            self.norm = nn.InstanceNorm2d(width)
        else:
            self.spectral = SpectralConv1d(width, width, n_modes_x)
            self.local = nn.Conv1d(width, width, kernel_size=1)
            self.norm = nn.InstanceNorm1d(width)
 
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: (batch, width, *spatial_dims)
        spectral layer
        """
        x_spectral = self.spectral(x)           # fourier layer (captures global structure)
        x_local = self.local(x)                 # local (pointwise)
        out = self.norm(x_spectral + x_local)   # add global and local operator 
        out = self.activation(out)
        return out + x                       #residual connection
    

class FourierNeuralOperator(nn.Module):
    """
    fourier neural operator for learning SPDE solution maps 
    
    what this model does: 
        - takes in fields (initial conditions + stochastic forcing + coordinates)
        - learns to output the solution field
        - uses Fourier layers to capture global physics

    FNO notes:
    learns by guessing a function, measuring error, and nudges parameters
    Stochastic Gradient Descent: the rule for how to nudge parameters

    architecture: 
        - lifting: takes raw physics variables and embed them into higher-dimensional feature space
        - fourier layers: stacks fourier layers to iteratively refine field, allowing buildup of nonlinear interactions
        - projection: takes learned feature representation to translate back into physical space
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_width: int,
                 n_fourier_layers: int, n_modes_x: int, n_modes_y: int = None):
        """
        Args:
            in_channels: number of input channels (initial condition + noise + coordinates)
            out_channels: number of output channels (solution components)
            hidden_width: width of hidden feature maps
            n_fourier_layers: number of stacked fourier layers
            n_modes_x: fourier modes in x direction
            n_modes_y: fourier modes in y direction (None for 1D problems)
        """
        super().__init__()
        self.is_2d = n_modes_y is not None
 
        # lifting
        if self.is_2d:
            self.lifting = nn.Sequential(
                nn.Conv2d(in_channels, hidden_width, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_width, hidden_width, kernel_size=1),
            )
        else:
            self.lifting = nn.Sequential(
                nn.Conv1d(in_channels, hidden_width, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden_width, hidden_width, kernel_size=1),
            )
 
        #fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_width, n_modes_x, n_modes_y)
            for _ in range(n_fourier_layers)
        ])
 
        #projection
        if self.is_2d:
            self.projection = nn.Sequential(
                nn.Conv2d(hidden_width, hidden_width, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_width, out_channels, kernel_size=1),
            )
        else:
            self.projection = nn.Sequential(
                nn.Conv1d(hidden_width, hidden_width, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden_width, out_channels, kernel_size=1),
            )
 
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dim_str = "2D" if self.is_2d else "1D"
        print(f"FNO ({dim_str}): {n_params:,} trainable parameters, "
              f"{n_fourier_layers} layers, width={hidden_width}")
        
    def forward(self, x):
        """
        foward pass through the full operator
            - takes input fields and maps them to a higher-dimensional space using a nonlinear map
                - creates a feature space where PDE mapping/complex transformations become easier to represent
            - evolve features with global (Fourier) operators, 
            - convert back into the solution field 
        
        Args: 
            x: (batch, in_channels, *spatial_dims)
            - 1D: (batch, C, nx)
            - 2D: (batch, C, nx, ny)
                batch - number of independent samples
                in_channels = C = number of input channels
                spatial_dims = grid over domain - 1D or 2D

        Shape change: 
            Lifting: ℝ^(batch x C_in x grid) -> ℝ^(batch x W x grid)
                - W = width
                for each spatial point s: 
                    h(s) = phi(z(s)), phi: ℝ^C_in -> ℝ^W
            
            Fourier layers loop: shape stays the same 

            Projection layer: (batch, W, *spatial) -> (batch, C_out, *spatial)
                
        Returns:
            (batch, out_channels, * spatial_dims) predicted solutions
        """

        h = self.lifting(x)         # (batch, width, *spatial)
        for layer in self.fourier_layers:
            h = layer(h)            # (batch, width, *spatial) 

        out = self.projection(h)    # (batch, width, *spatial)

        return out


if __name__ == "__main__":
    """=== test ==="""
 
    print("=== 1D FNO test ===")
    model_1d = FourierNeuralOperator(
        in_channels=3,      #IC + noise + x_coord
        out_channels=1,
        hidden_width=32,
        n_fourier_layers=4,
        n_modes_x=16,
    )
    x_1d = torch.randn(8, 3, 64)
    out_1d = model_1d(x_1d)
    print(f"  input: {x_1d.shape} -> output: {out_1d.shape}")
 
    print("\n=== 2D FNO test ===")
    model_2d = FourierNeuralOperator(
        in_channels=4,      #IC + noise + x_coord + y_coord
        out_channels=1,
        hidden_width=32,
        n_fourier_layers=4,
        n_modes_x=12,
        n_modes_y=12,
    )
    x_2d = torch.randn(4, 4, 32, 32)
    out_2d = model_2d(x_2d)
    print(f"  input: {x_2d.shape} -> output: {out_2d.shape}")
 
    print("\ndone.")