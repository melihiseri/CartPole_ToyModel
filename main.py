import sys
import math
import collections
from copy import deepcopy
from datetime import timedelta
import time

import gym
import torch

from .player import Player
from .utils import randomly_remove_element, set_seed

set_seed(1994)

def load_model(path: str) -> Player:
    """
    Load a saved Player instance and reinitialize the environment.

    Args:
        path (str): File path to the saved model.

    Returns:
        Player: The loaded Player instance with a fresh environment.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    
    loaded_player = torch.load(path)
    loaded_player.env = gym.make('CartPole-v1')
    print(f"Player instance loaded successfully from {path}")
    
    return loaded_player


def run_simulation(player_id: int, num_episodes: int):
    """
    Run a simulation for a single Player instance.

    Args:
        player_id (int): The ID of the player.
        num_episodes (int): The number of episodes to run.
    """
    env = gym.make('CartPole-v1')

    # Player Parameters
    T = 8
    C_memory_size = 64
    SA_memory_size = T * C_memory_size

    PlayerX = Player(
        env=env,
        T=T,
        stateaction_memory_size=SA_memory_size,
        max_memory=10,
        N_SVE_nets=8,
        SVE_size=256, # Neural network hidden size
        SVE_depth=2, # Neural network depth
        p_drop_SVE=0.0,  
        current_best_possible=15.0,
        state_value_control=False
    )

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        temperature = 1.0 # Fixed parameter.
        PlayerX.play_an_episode(temperature=temperature)

        # Record the total reward
        PlayerX.rewards_list.append(PlayerX.current_episode_reward)

        # Measures the overall performance in absolute sense for adjusting learning rate
        SF = (500.0 - PlayerX.current_best_possible) / 500.0
        # Measures the overall performance for regret analysis
        overall_performance = 2.5*(PlayerX.current_episode_reward / 500.0) ** 4
        # Measures the current performance for regret analysis
        current_performance = (PlayerX.current_episode_reward / PlayerX.current_best_possible)

        regret = current_performance + overall_performance

        # If episode memories are not full, overwrite best/worst regret
        if len(PlayerX.best_ones) != PlayerX.best_ones.maxlen:
            PlayerX.best_regret = 0.0
        if len(PlayerX.worst_ones) != PlayerX.worst_ones.maxlen:
            PlayerX.worst_regret = 1.0

        # Randomly remove memories
        randomly_remove_element(PlayerX.best_ones, 0.1)
        randomly_remove_element(PlayerX.worst_ones, 0.03)

        
        shift = 0.3 # Adjust regret values with a shift
        # Record the episode in best memories
        if regret + shift > PlayerX.best_regret:
            PlayerX.best_ones.append((deepcopy(PlayerX.stateaction_memory.memory), regret+shift))
            PlayerX.best_ones = collections.deque(
                sorted(PlayerX.best_ones, key=lambda x: x[1]), maxlen=PlayerX.best_ones.maxlen)
            PlayerX.best_regret = min(item[1] for item in PlayerX.best_ones)

        # Record the episode in worst memories
        if regret - shift < PlayerX.worst_regret:
            PlayerX.worst_ones.append((deepcopy(PlayerX.stateaction_memory.memory), max(regret-shift,0.2)))
            PlayerX.worst_ones = collections.deque(
                sorted(PlayerX.worst_ones, key=lambda x: x[1], reverse=True),
                maxlen=PlayerX.worst_ones.maxlen)
            PlayerX.worst_regret = max(item[1] for item in PlayerX.worst_ones)

            
        # Sleep Training (Past Episodes + time-consistency (DPP)) Every 6 Episodes
        if episode % 6 == 0:
            PlayerX.sleep_training(
                epoch=16,
                learning_rate=0.0001 * (SF**0.5),
                weight_decay=1e-4 * (SF**0.5),
                sampling_size=256,
                batch_size=16,
                alpha=0.0, beta=0.0,
                past_epoch=4, past_batch_size=128,
                temperature=temperature
            )
            
        # Clear Memory
        PlayerX.clear_episode_memory()
        
        # Print progress
        elapsed_time = time.time() - start_time
        avg_time_per_episode = elapsed_time / episode
        estimated_time_left = avg_time_per_episode * (num_episodes - episode)

        sys.stdout.write(
            f"\rPlayer {player_id}: {episode}/{num_episodes} episodes completed. "
            f"Time Left: {timedelta(seconds=int(estimated_time_left))}. "
            f"Current Possible: {int(PlayerX.current_best_possible)}"
        )
        sys.stdout.flush()

    # Save trained player model
    PlayerX.save_model(f'./model_PlayerID{player_id}_Episode{episode}.pth')
    print(f"Player {player_id} finished {num_episodes} episodes.")


if __name__ == '__main__':
    """
    Main script to run multiple players sequentially for a fixed number of episodes.

    Note: 
        Multiprocessing bottlenecks, potentially due to gym.
        Running sequentially was order of magnitude faster.
    """
    
    N_Player = 32  # Number of players
    num_episodes = 500  # Number of episodes per player

    print("Starting simulations...")

    for player_id in range(N_Player):
        print(f"\nStarting Player {player_id}...")
        run_simulation(player_id, num_episodes)

    print("\nAll players have completed their episodes.")
