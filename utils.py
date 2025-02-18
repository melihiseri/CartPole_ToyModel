import math
import random
import numpy as np
import torch
import torch.nn as nn

from typing import List, Dict, Literal, Tuple

def set_seed(seed: int = 1994) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed (int, optional): The seed value. Defaults to 1994.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Uncomment if training on GPU, ensures deterministic behavior for cuDNN
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    
def l4_regularization(model: nn.Module, lambda_l4: float) -> torch.Tensor:
    """
    Computes the L4 norm regularization penalty, normalized by the number of trainable parameters.

    Args:
        model (torch.nn.Module): The neural network model.
        lambda_l4 (float): Regularization strength.

    Returns:
        torch.Tensor: The L4 norm penalty term.
    """
    l4_penalty = sum(torch.sum(p.pow(4)) for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_params == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return (lambda_l4 / num_params) * l4_penalty


def compute_gradient_norm(network: nn.Module) -> float:
    """
    Compute the L2 norm of gradients for a given neural network.

    Args:
        network (torch.nn.Module): The neural network whose gradients will be measured.

    Returns:
        float: The L2 norm of all gradients.
    """
    grad_norms = [param.grad.norm(2).item() ** 2 for param in network.parameters() if param.grad is not None]
    return sum(grad_norms) ** 0.5 if grad_norms else 0.0



def compute_param_change(
    initial_params: List[Dict[str, torch.Tensor]], 
    networks: List[nn.Module], 
    aggregation: Literal["sum", "mean", "max"] = "mean"
) -> float:
    """
    Compute the aggregated L2 norm of parameter changes between initial and current model states.

    Args:
        initial_params (list of dict): List of initial state_dicts for each network.
        networks (list of nn.Module): List of networks to compare current parameters against initial ones.
        aggregation (str, optional): Aggregation method to summarize changes. 
            - 'sum': Returns the total sum of parameter changes.
            - 'mean': Returns the mean parameter change across networks.
            - 'max': Returns the maximum parameter change. 
            Defaults to 'mean'.

    Returns:
        float: Aggregated parameter change across all networks.
    """
    if len(initial_params) != len(networks):
        raise ValueError(f"Mismatch: {len(initial_params)} initial parameter sets vs {len(networks)} networks.")

    param_changes = []
    for init_param, network in zip(initial_params, networks):
        current_param = network.state_dict()
        
        # Compute L2 norm of parameter differences
        changes = torch.stack([
            torch.norm(current_param[key] - init_param[key])
            for key in init_param.keys()
        ])
        param_changes.append(changes.sum().item())
    
    if not param_changes:
        return 0.0

    if aggregation == "sum":
        return sum(param_changes)
    elif aggregation == "mean":
        return sum(param_changes) / len(param_changes)
    elif aggregation == "max":
        return max(param_changes)
    else:
        raise ValueError("Invalid aggregation method. Choose 'sum', 'mean', or 'max'.")



def map_to_range(
    x: torch.Tensor, 
    domains: torch.Tensor, 
    target_range: tuple[float, float] = (0.36, 0.86)
) -> torch.Tensor:
    """
    Maps each feature in `x` from its original domain to a target range using a linear transformation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, n_features).
        domains (torch.Tensor): A tensor of shape (n_features, 2) where each row contains (min, max) 
            values defining the original range of each feature.
        target_range (tuple of float, optional): The (min_target, max_target) range to map values into.
            Defaults to (0.36, 0.86).

    Returns:
        torch.Tensor: A transformed tensor of the same shape as `x`, mapped to the target range.
    """
    min_target, max_target = target_range

    domain_mins = domains[:, 0]
    domain_maxs = domains[:, 1]

    scaled_x = (x - domain_mins) / (domain_maxs - domain_mins)
    return scaled_x * (max_target - min_target) + min_target



def calculate_regret_series(
    initial_regret: float, 
    length: int, 
    sampling_type: str, 
    decay_factor: float = 0.95
) -> torch.Tensor:
    """
    Generate a regret series with exponential decay.

    Args:
        initial_regret (float): Initial regret value.
        length (int): Number of values in the series.
        sampling_type (str): Determines the order of regrets ('recent' or 'preceding').
        decay_factor (float, optional): Multiplicative decay rate (default: 0.95).

    Returns:
        torch.Tensor: A tensor of regret values.
    """
    indices = torch.arange(length, dtype=torch.float32)
    regrets = initial_regret * (decay_factor ** indices) + (1 - decay_factor ** indices)

    return regrets.flip(0) if sampling_type == "preceding" else regrets



def sample_indices(
    memory_length: int, 
    batch_size: int, 
    sampling_type: str, 
    alpha: float = 1.0, 
    exp_start: float = 0.005,
    recent_window_size: int = None
) -> torch.Tensor:
    """
    Samples indices from a hybrid exponential-uniform distribution.

    Args:
        memory_length (int): The total number of available indices.
        batch_size (int): The number of indices to sample.
        sampling_type (str): 
            - 'recent': Favors recent indices.
            - 'preceding': Favors older indices.
        alpha (float, optional): Weight between exponential and uniform sampling.
            - `alpha = 1.0` Fully exponential sampling.
            - `alpha = 0.0` Fully uniform sampling.
        exp_start (float, optional): Lower bound for the exponential distribution. Default is 0.005.
        recent_window_size (int, optional): The number of recent indices to sample from when sampling_type is "recent".

    Returns:
        torch.Tensor: A tensor of sampled indices.
    """

    if recent_window_size is not None:
        recent_window_size = min(recent_window_size, memory_length)
    
    # If "recent" sampling, restrict the memory range
    if sampling_type == "recent" and recent_window_size is not None:
        start_index = max(0, memory_length - recent_window_size)
        memory_length = recent_window_size
    else:
        start_index = 0  # Sample from the full range

    # Create an exponential distribution
    exponential_distribution = torch.logspace(math.log10(exp_start), math.log10(1.0), steps=memory_length)

    # Reverse if sampling type is 'preceding'
    if sampling_type == "preceding":
        exponential_distribution = exponential_distribution.flip(dims=(0,))

    # Normalize the exponential distribution
    exponential_distribution /= exponential_distribution.sum()

    # Create a uniform distribution
    uniform_distribution = torch.full((memory_length,), 1.0 / memory_length)

    # Mix exponential and uniform distributions
    combined_distribution = alpha * exponential_distribution + (1 - alpha) * uniform_distribution
    combined_distribution /= combined_distribution.sum()

    # Sample indices
    sampled_indices = torch.multinomial(combined_distribution, batch_size, replacement=True)

    # Adjust indices back to the original memory space if restricted
    sampled_indices += start_index  

    return sampled_indices



def analyze_states_and_regrets(
    state_regret_pairs: List[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], float]],
    batch_size: int, 
    sampling_size: int, 
    sampling_type: str, 
    decay_factor: float, 
    alpha: float, 
    exp_start: float = 0.005
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analyzes state-action memories and corresponding regret values to extract the most informative states.
    Uses kernel-based similarity and weighted regret calculations.

    Args:
        state_regret_pairs (List[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], float]]): 
            Each tuple contains a sequence of (state-probability pairs) and an initial regret value.
        batch_size (int): Number of top states to return based on kernel similarity.
        sampling_size (int): Number of samples to extract from each state memory.
        sampling_type (str): Defines the sampling strategy ('preceding' or 'recent').
        decay_factor (float): Decay factor for computing the regret series.
        alpha (float): Parameter used in the sampling function.
        exp_start (float, optional): Small starting value for exponential-based sampling. Defaults to 0.005.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - A tensor containing the top selected state representations.
            - A tensor containing the corresponding weighted average regret values.
    """
    
    all_states = []
    all_regrets = []

    for state_action_memory, initial_regret in state_regret_pairs:
        states = torch.stack([state[0] for state in state_action_memory])
        regrets = calculate_regret_series(initial_regret, len(states), sampling_type, decay_factor)

        # Default sampling size if None
        sampling_size = sampling_size or len(states) // 2

        # Sample indices
        indices = sample_indices(
            memory_length=len(states),
            batch_size=sampling_size,
            sampling_type=sampling_type,
            alpha=alpha,
            exp_start=exp_start,
            recent_window_size=20
        )

        # Collect sampled states and regrets
        all_states.append(states[indices])
        all_regrets.append(regrets[indices])

    # Concatenate all sampled states and regrets
    all_states_tensor = torch.cat(all_states, dim=0)
    all_regrets_tensor = torch.cat(all_regrets, dim=0)

    # Compute pairwise squared distances
    pairwise_diff = all_states_tensor.unsqueeze(0) - all_states_tensor.unsqueeze(1)
    pairwise_distances = torch.sum(pairwise_diff ** 2, dim=-1)

    # Apply kernel function
    bandwidth = torch.median(pairwise_distances)
    kernel_values = torch.exp(-pairwise_distances / bandwidth)

    # Select top states based on kernel similarity
    kernel_sums = torch.sum(kernel_values, dim=1)
    batch_size = min(batch_size, len(kernel_sums))
    top_indices = torch.topk(kernel_sums, k=batch_size, largest=True).indices

    # Compute weighted average regrets for selected states
    top_states = all_states_tensor[top_indices]
    kernel_weights = kernel_values[top_indices]
    weighted_regrets = torch.sum(kernel_weights * all_regrets_tensor, dim=1) / torch.sum(kernel_weights, dim=1)

    return top_states, weighted_regrets



def classify_cartpole_states(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify CartPole states into good and bad based on predefined criteria.
    Can be used for evaluating SVE networks.

    Args:
        states (torch.Tensor): Tensor of shape [batch_size, 4], where each row is [x, x_dot, theta, theta_dot].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors containing good and bad states.

    Notes:
        - Good states: Cart near the center, low velocity, nearly upright pole.
        - Bad states: Cart near boundaries, high velocity, significantly deviated pole.
    """
    # Extract individual components
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

    # Define good state conditions
    good_mask = (
    (torch.abs(x) < 1.0)
        & (torch.abs(x_dot) < 0.5)
        & (torch.abs(theta) < 0.05)
        & (torch.abs(theta_dot) < 1.0)
    )

    bad_mask = (
        (torch.abs(x) > 2.4)
        | (torch.abs(x_dot) > 2.0)
        | (torch.abs(theta) > 0.16)
        | (torch.abs(theta_dot) > 2.5)
    )

    good_states = states[good_mask]
    bad_states = states[bad_mask]

    if good_states.numel() == 0:
        good_states = torch.empty((0, states.shape[1]), device=states.device)
    if bad_states.numel() == 0:
        bad_states = torch.empty((0, states.shape[1]), device=states.device)

    return good_states, bad_states



def generate_balanced_cartpole_states(num_states=1000):
    """
    Generate a tensor of balanced CartPole states (half good, half bad).
    Can be used for evaluating SVE networks.

    Args:
        num_states (int): Total number of states to generate (should be even).

    Returns:
        torch.Tensor: A tensor containing good states followed by bad states.
    """
    assert num_states % 2 == 0, "Number of states must be even to balance good and bad."

    good_states, bad_states = [], []
    target_per_class = num_states // 2

    while len(good_states) < target_per_class or len(bad_states) < target_per_class:
        random_states = (torch.rand((100, 4)) * torch.tensor([5.0, 6.0, 0.5, 6.0])) - torch.tensor([2.5, 3.0, 0.25, 3.0])

        good_batch, bad_batch = classify_cartpole_states(random_states)

        # Add to the respective lists while ensuring we don't exceed the targets
        good_states.extend(good_batch[:target_per_class - len(good_states)])
        bad_states.extend(bad_batch[:target_per_class - len(bad_states)])

    balanced_states = torch.cat([
        torch.stack(good_states),
        torch.stack(bad_states)
    ], dim=0)

    assert len(good_states) == len(bad_states), \
        f"Number of good states ({len(good_states)}) and bad states ({len(bad_states)}) are not equal."

    return balanced_states





def generate_strategies(T: int) -> torch.Tensor:
    """
    Generate all possible 2^T binary strategies of length T.

    Args:
        T (int): Length of each strategy.

    Returns:
        torch.Tensor: A tensor of shape [2^T, T], where each row is a strategy.
    """
    
    num_strategies = 2 ** T  # Total number of strategies
    indices = torch.arange(num_strategies, dtype=torch.int32)

    # Convert to binary representation using bitwise operations
    strategies = (indices[:, None] >> torch.arange(T - 1, -1, -1)) & 1

    return strategies.to(dtype=torch.float32)



def randomly_remove_element(dq, epsilon):
    """
    Randomly removes an element from a deque with a given probability.

    Args:
        dq (collections.deque): The deque from which to remove an element.
        epsilon (float): Probability of removing an element.
    """

    if dq and random.random() < epsilon:
        random_index = random.randint(0, len(dq) - 1) 
        dq.rotate(-random_index)  # Rotate target index to front
        dq.popleft()  # Remove the front element
        dq.rotate(random_index)  # Rotate back
