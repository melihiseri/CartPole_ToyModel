import torch
from typing import Tuple

def cartpole_transition(
    states: torch.Tensor, 
    actions: torch.Tensor, 
    dt: float = 0.02, 
    gravity: float = 9.8, 
    mass_cart: float = 1.0, 
    mass_pole: float = 0.1, 
    length: float = 0.5, 
    force_mag: float = 10.0
) -> torch.Tensor:
    """
    Computes the next state of the CartPole system for a batch of states and actions using PyTorch.

    Args:
        states (torch.Tensor): A tensor of shape [batch_size, 4] where each row represents 
            [x, x_dot, theta, theta_dot].
        actions (torch.Tensor): A tensor of shape [batch_size, 1] containing discrete actions (0 or 1).
        dt (float, optional): Time step duration. Default is 0.02.
        gravity (float, optional): Gravitational acceleration. Default is 9.8.
        mass_cart (float, optional): Mass of the cart. Default is 1.0.
        mass_pole (float, optional): Mass of the pole. Default is 0.1.
        length (float, optional): Half-length of the pole. Default is 0.5.
        force_mag (float, optional): Magnitude of the applied force. Default is 10.0.

    Returns:
        torch.Tensor: The next states of shape [batch_size, 4].
    """
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

    # Apply force based on action
    forces = torch.where(actions.flatten() == 1, force_mag, -force_mag)

    total_mass = mass_cart + mass_pole
    pole_mass_length = mass_pole * length
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    # Compute acceleration components
    temp = (forces + pole_mass_length * theta_dot**2 * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - mass_pole * costheta**2 / total_mass)
    )
    x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass

    # Compute next state values
    x_next = x + dt * x_dot
    x_dot_next = x_dot + dt * x_acc
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * theta_acc

    return torch.stack([x_next, x_dot_next, theta_next, theta_dot_next], dim=1)


def exact_transition(
    state: torch.Tensor, 
    control: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes exact state transitions and endpoints for given initial states and control sequences.

    Args:
        state (torch.Tensor): A tensor of shape [batch_size_states, state_dim] representing the 
            initial states.
        control (torch.Tensor): A tensor of shape [batch_size_control, T] containing control sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - `paths`: Tensor of shape [batch_size_states, batch_size_control, T + 1, state_dim] 
              (or [batch_size_control, T + 1, state_dim] if `batch_size_states == 1`), representing 
              the full trajectory.
            - `endpoints`: Tensor of shape [batch_size_states, batch_size_control, state_dim] 
              (or [batch_size_control, state_dim] if `batch_size_states == 1`), representing the 
              final states after applying the control sequence.
    """
    batch_size_states = state.shape[0]
    batch_size_control = control.shape[0]
    T = control.shape[1]
    state_dim = state.shape[1]

    # Expand state to match batch_size_control if necessary
    if batch_size_states == 1 and batch_size_control > 1:
        state = state.repeat(batch_size_control, 1)  # Avoids unwanted tensor sharing

    # Initialize path and endpoint tensors
    if batch_size_states > 1:
        path_tensor = torch.zeros(batch_size_states, batch_size_control, T + 1, state_dim, device=state.device)
        endpoint_tensor = torch.zeros(batch_size_states, batch_size_control, state_dim, device=state.device)
    else:
        path_tensor = torch.zeros(batch_size_control, T + 1, state_dim, device=state.device)
        endpoint_tensor = torch.zeros(batch_size_control, state_dim, device=state.device)

    for i in range(batch_size_states):
        path = [state[i].unsqueeze(0).expand(batch_size_control, -1)]  # Shape: [batch_size_control, state_dim]
        current_state = path[0]

        for t in range(T):
            action = control[:, t].unsqueeze(1)  # Extract action at time t
            current_state = cartpole_transition(current_state, action)  # Compute next state
            path.append(current_state)

        if batch_size_states > 1:
            path_tensor[i] = torch.stack(path, dim=1)  # Shape: [batch_size_control, T + 1, state_dim]
            endpoint_tensor[i] = current_state  # Shape: [batch_size_control, state_dim]
        else:
            path_tensor = torch.stack(path, dim=1)  # Shape: [batch_size_control, T + 1, state_dim]
            endpoint_tensor = current_state  # Shape: [batch_size_control, state_dim]

    return path_tensor, endpoint_tensor
