import torch
import torch.nn as nn

from .utils import map_to_range


class StateValueEstimator(nn.Module):
    """
    A neural network model that estimates state values for the CartPole environment.

    The model normalizes the input states, passes them through multiple hidden layers 
    with dropout and LeakyReLU activations, and outputs a scaled value prediction.
    """
    def __init__(
            self,
            input_dim: int = 4,
            hidden_dim: int = 64,
            output_dim: int = 1,
            depth: int = 4,
            dropout_prob: float = 0.1,
            negative_slope: float = 0.01  
    ):
        """
        Initializes the StateValueEstimator network.

        Args:
            input_dim (int): Number of input features (state dimensions). Default is 4.
            hidden_dim (int): Number of units in hidden layers. Default is 64.
            output_dim (int): Number of output features. Default is 1.
            depth (int): Number of hidden layers. Default is 4.
            dropout_prob (float): Dropout probability. Default is 0.1.
            negative_slope (float): LeakyReLU negative slope. Default is 0.01.
        """
        super(StateValueEstimator, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.negative_slope = negative_slope
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) 
            for i in range(depth)
        ])

        # Final layer
        self.fc_last = nn.Linear(hidden_dim, output_dim)

        # Store state bounds for normalization
        self.register_buffer("state_bounds", torch.tensor(
            [[-4.8, 4.8], [-3.0, 3.0], [-0.418, 0.418], [-2.0, 2.0]], dtype=torch.float32
        ))

    def forward(self, state, scale: float = 1.0):
        """
        Forward pass of the network.

        Args:
            state (torch.Tensor): Input tensor of shape [batch_size, 4].
            scale (float, optional): Scaling factor for the output. Default is 1.0.

        Returns:
            torch.Tensor: Scaled state value prediction in the range [0, 100].
        """
        
        # Normalize state to (-2, 2)
        x = map_to_range(state, self.state_bounds, target_range=(-2.0, 2.0))
        
        # Hidden layers | dropout -> linear -> leaky relu
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)

        # Final output layer
        x = self.fc_last(x)
        
        # Scale output to [0, 100]
        return 50.0 * (torch.tanh(x * scale) + 1)



    def apply_noise_to_parameters(self, noise_level: float = 1e-6):
        """
        Adds Gaussian noise to the model parameters.
        
        Args:
            noise_level (float, optional): Standard deviation of the Gaussian noise to apply.
        """
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * noise_level)
