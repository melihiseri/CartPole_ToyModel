import collections
import numpy as np
import torch

from typing import Tuple, List

from .utils import sample_indices


class Memory:
    """
    A memory buffer to store and retrieve (space, observation) pairs.

    This class maintains a fixed-size buffer using a deque, allowing
    addition and sampling of (space, observation) pairs.
    """

    def __init__(self, memory_size: int) -> None:
        """
        Initialize a memory buffer with a fixed maximum size.

        Args:
            memory_size (int): Maximum number of (space, observation) pairs to store.
        """
        self.memory_size = memory_size
        self.memory = collections.deque(maxlen=memory_size)

    def remember(self, space: torch.Tensor, observation: torch.Tensor) -> None:
        """
        Store a (space, observation) pair in memory.

        Args:
            space (torch.Tensor | np.ndarray | list | tuple): The input or state representation.
            observation (torch.Tensor | np.ndarray | list | tuple): The corresponding output or observation.

        Notes:
            - Both `space` and `observation` are converted to `torch.Tensor` if needed.
            - They are flattened to 1D tensors before storage.
        """
        if isinstance(space, np.ndarray):
            space = torch.tensor(space, dtype=torch.float32)
        elif isinstance(space, (list, tuple)):
            space = torch.tensor(space, dtype=torch.float32)

        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32)
        elif isinstance(observation, (list, tuple)):
            observation = torch.tensor(observation, dtype=torch.float32)

        if space.dim() > 1:
            space = space.view(-1)
        if observation.dim() > 1:
            observation = observation.view(-1)
        
        self.memory.append((space, observation))


    def get_memory(
        self, 
        batch_size: int, 
        sampling_type: str, 
        alpha: float, 
        add_randomness: bool = False, 
        noise_level: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of (space, observation) pairs from memory.

        Args:
            batch_size (int): Number of samples to retrieve.
            sampling_type (str): Sampling strategy ('recent' or 'preceding').
            alpha (float): Weighting factor between exponential and uniform sampling.
                - `alpha = 1.0`: Fully exponential distribution.
                - `alpha = 0.0`: Fully uniform distribution.
            add_randomness (bool, optional): If True, adds Gaussian noise to sampled spaces. Default is False.
            noise_level (float, optional): Standard deviation of the Gaussian noise. Default is 1e-6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - `space_samples`: Sampled space components of shape `(batch_size, ...)`.
                - `observation_samples`: Corresponding sampled observation components.

        Raises:
            ValueError: If memory is empty.

        """
        if len(self.memory) == 0:
            raise ValueError("No memory to sample from.")

        memory_list = list(self.memory)
        all_spaces, all_observations = zip(*memory_list)
        all_spaces = torch.stack(all_spaces)
        all_observations = torch.stack(all_observations)

        memory_indices = sample_indices(
            memory_length=len(self.memory),
            batch_size=batch_size, 
            sampling_type=sampling_type, 
            alpha=alpha
        )

        if add_randomness:
            space_samples = torch.stack([
                torch.normal(mean=all_spaces[idx], std=noise_level) for idx in memory_indices
            ])
        else:
            space_samples = all_spaces[memory_indices]

        observation_samples = all_observations[memory_indices]
        return space_samples, observation_samples

    def get_raw_memory(self, batch_size: int, sampling_type: str, alpha: float) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Retrieve indices and raw memory content for debugging or further processing.

        Args:
            batch_size (int): Number of samples to retrieve.
            sampling_type (str): Sampling strategy to use.
            alpha (float): Parameter for sampling distribution.

        Returns:
            Tuple:
                - `memory_indices` (List[int]): Indices of the sampled memory.
                - `all_spaces` (torch.Tensor): All space components in memory.
                - `all_observations` (torch.Tensor): All observation components in memory.

        Raises:
            ValueError: If memory is empty.
        """
        if len(self.memory) == 0:
            raise ValueError("No memory to sample from.")

        memory_list = list(self.memory)
        all_spaces, all_observations = zip(*memory_list)
        all_spaces = torch.stack(all_spaces)
        all_observations = torch.stack(all_observations)

        memory_indices = sample_indices(
            memory_length=len(self.memory),
            batch_size=batch_size, 
            sampling_type=sampling_type, 
            alpha=alpha
        )

        return memory_indices.tolist(), all_spaces, all_observations

    def view_memory(self) -> None:
        """
        Print the current contents of the memory buffer for debugging purposes.
        """
        print("Memory contents:")
        for i, (space, observation) in enumerate(self.memory):
            print(f"Entry {i}:")
            print(f"  Space: {space}")
            print(f"  Observation: {observation}")

    def clear(self) -> None:
        """
        Clear all entries in the memory buffer.
        """
        self.memory = collections.deque(maxlen=self.memory_size)
