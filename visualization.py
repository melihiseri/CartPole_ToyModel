import os
import torch
import matplotlib.pyplot as plt
from typing import List

from .player import Player


def load_rewards(model_paths: List[str]) -> List[List[float]]:
    """
    Load rewards from saved Player models.

    Args:
        model_paths (List[str]): List of paths to saved Player models.

    Returns:
        List[List[float]]: A list where each element is a list of episode rewards for a Player.
    """
    all_rewards = []

    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model file {path} not found. Skipping...")
            continue

        player = torch.load(path, weights_only=False)  
        all_rewards.append(player.rewards_list)

    return all_rewards



def compute_moving_average(rewards: List[float], window_size: int = 100) -> List[float]:
    """
    Compute the moving average of rewards over a given window size.

    Args:
        rewards (List[float]): List of episode rewards.
        window_size (int, optional): The number of games to compute the average over. Default is 100.

    Returns:
        List[float]: The moving average rewards list.
    """
    moving_avg = []
    for i in range(len(rewards)):
        window = rewards[max(0, i - window_size + 1): i + 1]
        moving_avg.append(sum(window) / len(window))  # Compute average over the available window

    return moving_avg


def plot_rewards(model_paths: List[str], output_file: str = "rewards_plot") -> None:
    """
    Plot all players episode rewards on a single figure.

    Args:
        model_paths (List[str]): List of paths to saved Player models.
        output_file (str, optional): Path to save the plot. Default is "rewards_plot.png".
    """
    all_rewards = load_rewards(model_paths)


    
    fig, ax = plt.subplots(figsize=(7,3.5), dpi=600)

    for i, rewards in enumerate(all_rewards):
        if not rewards:
            continue  # Skip empty reward lists
        color = plt.gca()._get_lines.get_next_color()
        
        moving_avg_rewards = compute_moving_average(rewards, window_size=100)

        # Plot raw rewards
        plt.plot(range(1, len(rewards) + 1), rewards, alpha=0.5, marker='o', ms=1.0,
                 linestyle=" ", label=f'Player {i+1} (Raw)', color=color)

        # Plot moving average
        plt.plot(range(1, len(moving_avg_rewards) + 1), moving_avg_rewards, linewidth=1.0,
                 marker='*', ms=1.0, linestyle="-", label=f'Player {i+1} (Avg)', color=color)

    # ax.set_xlabel("Number of Games", fontsize=10)
    # ax.set_ylabel("Total Episode Reward", fontsize=10)
    # ax.set_title("Player Episode Rewards with Moving Average", fontsize=10)

    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    plt.savefig(output_file+'.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(output_file+'.png', format="png", bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {output_file}.pdf .png")


if __name__ == "__main__":
    """
    Main script to load saved player models and visualize rewards.
    """
    
    episode_number = 500
    N_Player = 32

    selected_players = None

    # seed = 1994:
    # selected_players = [3, 0, 9, 11, 12, 16, 19, 23, 21, 24]
    
    # non-seeded ones:
    # selected_players = [3, 4, 13, 0, 1, 6, 11, 14] 

    
    # Generate file paths based on the saved naming convention
    if selected_players == None:
        model_paths = [f"./model_PlayerID{player_id}_Episode{episode_number}.pth" for player_id in range(N_Player)]
    else:
        model_paths = [f"./model_PlayerID{player_id}_Episode{episode_number}.pth" for player_id in selected_players]

    plot_rewards(model_paths)
