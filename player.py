import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .memory import Memory
from .networks import StateValueEstimator
from .transition import exact_transition
from .utils import (
    generate_strategies,
    l4_regularization,
    compute_gradient_norm,
    analyze_states_and_regrets,
    generate_balanced_cartpole_states,
    set_seed
)



class Player:
    def __init__(self,
                 env,
                 T: int,
                 stateaction_memory_size: int,
                 max_memory: int,
                 N_SVE_nets: int,
                 SVE_size: int,
                 SVE_depth: int,
                 p_drop_SVE: float,
                 current_best_possible: float,
                 state_value_control: bool = False):
        """
        Initializes the Player agent for CartPole learning.

        Args:
            env: The Gym environment instance.
            T (int): Horizon
            stateaction_memory_size (int): Maximum memory size per episode.
            max_memory (int):   Maximum number of stored episodes.
            N_SVE_nets (int):   Number of State Value Estimator (SVE) networks.
            SVE_size (int):     Hidden layer size for SVE networks.
            SVE_depth (int):    Depth (number of layers) of SVE networks.
            p_drop_SVE (float): Dropout probability for SVE networks.
            current_best_possible (float): Expected best possible episode reward.
            state_value_control (bool, optional): Whether to initialize state value control.
        """

        set_seed(1994)
        
        self.env = env
        self.T = int(T)
        self.all_controls = generate_strategies(self.T)
        
        self.state_bounds = [(-4.8, 4.8), (-3.0, 3.0),
                             (-0.418, 0.418), (-2.0, 2.0)]

        #===
        
        self.SVE_size = SVE_size
                
        self.SVE_networks = [StateValueEstimator(input_dim=4,
                                                 hidden_dim=SVE_size,
                                                 output_dim=1,
                                                 depth = SVE_depth,
                                                 dropout_prob=p_drop_SVE)
                             for _ in range(N_SVE_nets)]
        [sve_net.train() for sve_net in self.SVE_networks]
        
        self.SVE_optimizers = [
            torch.optim.AdamW(sve_net.parameters(), lr=0.001, weight_decay=1e-4)
            for sve_net in self.SVE_networks
        ]

        #===
        
        self.stateaction_memory_size = stateaction_memory_size        
        self.stateaction_memory = Memory(memory_size=stateaction_memory_size)
        
        self.past_states = collections.deque(maxlen=512)

        # Recording the best and worst episodes
        self.best_ones = collections.deque(maxlen=max_memory) 
        self.best_regret = 0.0
        self.worst_ones = collections.deque(maxlen=max_memory*2)
        self.worst_regret = 1.0

        
        self.current_episode_reward = 0
        self.current_best_possible = current_best_possible
        self.rewards_list = []

        
        if state_value_control:
            self.check_states = generate_balanced_cartpole_states(1000)

        
        

    def play_an_episode(self, temperature: float = 1.0):
        """
        Runs a single episode of CartPole using learned State Value Estimators.

        Args:
            temperature (float, optional): Softmax temperature parameter for action selection.
        """
        self.current_episode_reward = 0
        observation, _ = self.env.reset()
        observation = torch.tensor(observation).unsqueeze(0)
        
        done = False
        counter = 1        # Counting the number of steps
        action_counter = 0 # Index of current action
        
        while not done:
            if (counter - 1) % self.T == 0:
                """ Sample control for the next self.T steps """
                action_counter = 0

                exact_terminals = exact_transition(observation, self.all_controls)[1]

                all_probabilities = []

                for sve_net in self.SVE_networks:
                    current_value = sve_net(observation)

                    differences = sve_net(exact_terminals) - current_value

                    probabilities = torch.softmax(differences / temperature, dim=0)
                    all_probabilities.append(probabilities.squeeze(-1))

                all_probabilities = torch.stack(all_probabilities) # Shape: [N_SVE_nets, 2^T]

                # Uniform distribution over value networks
                induced_distribution = all_probabilities.mean(dim=0)  # Shape: [2^T]

                sampled_control_index = torch.multinomial(induced_distribution, 1, replacement=True)
                sampled_control = self.all_controls[sampled_control_index].squeeze(0)

            a = sampled_control[action_counter]

            
            # Records the probabilities assigned to the sampled action
            # by each network. Induced probabilities are not used, but
            # can be to identify which networks yielded better results
            matching_indices = (self.all_controls[:, action_counter] == a).nonzero(as_tuple=True)[0]
            induced_probabilities = all_probabilities[:, matching_indices].sum(dim=1) # [N_SVE_nets, 1]
            
            prev_state = observation.clone().detach()

            # Record            
            self.past_states.append(prev_state.clone().detach().squeeze())
            self.stateaction_memory.remember(prev_state.clone().detach(), induced_probabilities.clone().detach())
            
            # Take a step
            observation, reward, done, truncated, info = self.env.step(int(a.item()))
            observation = torch.tensor(observation).unsqueeze(0)
            
            self.current_episode_reward += reward
            done = done or truncated

            counter += 1
            action_counter += 1 

        # Internal moving average for adjusting optimizer rates
        weight = 0.1 if self.current_episode_reward > 0 else 0.03
        self.current_best_possible = (1-weight)* self.current_best_possible + weight*self.current_episode_reward
        


    # ==== Sleep Training
    
    def sleep_training(self,
                       epoch: int,
                       learning_rate: float,
                       weight_decay: float,
                       sampling_size: int,
                       batch_size: int,
                       alpha: float = 0.0,
                       beta: float = 0.05,
                       past_epoch: int = 4,
                       past_batch_size: int = 32,
                       temperature: float = 1.0):
        """
        Trains SVEs on past memories and for time-consistency.

        Args:
            epoch (int): Number of training epochs for SVE networks.
            learning_rate (float): Learning rate for training.
            weight_decay (float): Weight decay for regularization.
            sampling_size (int): Number of samples to use from memory.
            batch_size (int): Batch size for training.
            alpha (float, optional): Weighting factor for regret analysis. Defaults to 0.0.
            beta (float, optional): Weight decay factor for regret update. Defaults to 0.05.
            past_epoch (int, optional): Number of epochs for DPP training. Defaults to 4.
            past_batch_size (int, optional): Batch size for DPP training. Defaults to 32.
            temperature (float, optional): Temperature parameter for softmax calculations. Defaults to 1.0.
        """

        
        """ Collecting the States and Regrets from best and worst episodes """
        # Analyze best_ones
        if self.best_ones:
            best_states_tensor, best_regrets_tensor = analyze_states_and_regrets(
                state_regret_pairs=self.best_ones,
                batch_size=sampling_size,
                sampling_size=None, # Samples half of the memory
                sampling_type='preceding',
                decay_factor=0.999,
                alpha=1.0,
                exp_start=0.05
            )

        # Analyze worst_ones
        if self.worst_ones:
            worst_states_tensor, worst_regrets_tensor = analyze_states_and_regrets(
                state_regret_pairs=self.worst_ones,
                batch_size=sampling_size,
                sampling_size=10, # Only sample 10 states from worst ones
                sampling_type='recent',
                decay_factor=0.95,
                alpha=1.0,
                exp_start=0.005
            )

            
        # Concatenate them
        if (self.best_ones and self.worst_ones):
            top_states_tensor = torch.cat([best_states_tensor, worst_states_tensor])
            average_regrets_tensor = torch.cat([best_regrets_tensor, worst_regrets_tensor])
        elif self.best_ones:
            top_states_tensor = best_states_tensor
            average_regrets_tensor = best_regrets_tensor
        elif self.worst_ones:
            top_states_tensor = worst_states_tensor
            average_regrets_tensor = worst_regrets_tensor

        
        # Train the networks
        self.sleeptrain_statevalue_networks(
            states=top_states_tensor, regrets=average_regrets_tensor,
            epoch=epoch, learning_rate=learning_rate, weight_decay=weight_decay,
            batch_size=batch_size,
            past_epoch=4, past_batch_size=32, temperature=temperature
        )
        

        for i in range(len(self.best_ones)):
            d1, regret = self.best_ones[i]
            self.best_ones[i] = (d1, beta + (1 - beta) * regret)

        for i in range(len(self.worst_ones)):
            d1, regret = self.worst_ones[i]
            self.worst_ones[i] = (d1, beta + (1 - beta) * regret)

        self.clear_episode_memory()
        

    def sleeptrain_statevalue_networks(self,
                                       states: torch.Tensor,
                                       regrets: torch.Tensor,
                                       epoch: int,
                                       learning_rate: float,
                                       weight_decay: float,
                                       batch_size: int,
                                       past_epoch: int = 4,
                                       past_batch_size: int = 32,
                                       temperature: float = 1.0):
        """
        Trains the State Value Estimators (SVE) using stored state and regret data.

        Args:
            states (torch.Tensor): Tensor of sampled states from memory.
            regrets (torch.Tensor): Corresponding regret values for states.
            epoch (int): Number of training epochs.
            learning_rate (float): Learning rate for training.
            weight_decay (float): Weight decay for regularization.
            batch_size (int): Batch size for training.
            past_epoch (int, optional): Number of epochs for DPP training. Defaults to 4.
            past_batch_size (int, optional): Batch size for DPP training. Defaults to 32.
            temperature (float, optional): Temperature parameter for probability weighting. Defaults to 1.0.
        """


        

        """ First part of training is learning from memory """
        # Transform regrets into targets in [0,100]
        targets = 1 / (1 + torch.exp(-14.0 * (regrets - 1.0)))
        targets = (targets * 100.0).view(-1, 1)
        bins = [0, 35, 65, 100]  # Define bins for target values
        bin_indices = np.digitize(targets, bins)  # Assign each target to a bin

        # Update learning rates and weight decay for all optimizers
        for optimizer in self.SVE_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['weight_decay'] = weight_decay
                
        # Train each network
        for sve_net, optimizer in zip(self.SVE_networks, self.SVE_optimizers):
            for ep in range(epoch):
                subbatch_indices = []
                # Ensure roughly equal contribution from each bin
                for b in range(1, len(bins)):
                    bin_mask = bin_indices == b  # Mask for current bin
                    bin_pool = np.where(bin_mask)[0]  # Indices of samples in this bin
                    if len(bin_pool) > 0:
                        num_samples = batch_size // len(bins)
                        sampled_indices = np.random.choice(bin_pool, num_samples, replace=True)
                        subbatch_indices.extend(sampled_indices)

                # If we need more samples to fill the batch, sample remaining randomly
                if len(subbatch_indices) < batch_size:
                    remaining_indices = np.setdiff1d(range(len(targets)), subbatch_indices)
                    extra_samples = np.random.choice(remaining_indices, batch_size - len(subbatch_indices), replace=True)
                    subbatch_indices.extend(extra_samples)

                # Fetch the states and targets for the batch
                subbatch_states = states[subbatch_indices]
                subbatch_targets = targets[subbatch_indices]
                
                outputs = sve_net(subbatch_states)
                loss = torch.mean((outputs - subbatch_targets)**4)
                total_loss = loss + l4_regularization(sve_net, 1e-3)

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(sve_net.parameters(), max_norm=5.0)
                optimizer.step()


        """ Second part of training is for time-consistency (DPP) """
        past_states = torch.stack(list(self.past_states))
        for sve_net, optimizer in zip(self.SVE_networks, self.SVE_optimizers):
            for ep in range(past_epoch):

                subset_indices = torch.randperm(past_states.size(0))[:past_batch_size]
                sampled_states = past_states[subset_indices]  # Shape: [subset_size, state_dim]

                exact_terminals = exact_transition(sampled_states, self.all_controls)[1]  # Shape: [subset_size, batch_size_control, state_dim]

                current_value = sve_net(sampled_states)  # Shape: [subset_size, 1]

                flattened_terminals = exact_terminals.view(-1, exact_terminals.shape[-1])  # Shape: [(subset_size * batch_size_control), state_dim]
                flattened_values = sve_net(flattened_terminals)  # Shape: [(subset_size * batch_size_control), 1]
                reshaped_values = flattened_values.view(exact_terminals.shape[0], exact_terminals.shape[1], -1)  # Shape: [subset_size, batch_size_control, 1]

                expanded_current_value = current_value.unsqueeze(1)  # Shape: [subset_size, 1, 1]
                differences = reshaped_values - expanded_current_value  # Shape: [subset_size, batch_size_control, 1]

                probabilities = torch.softmax(differences / temperature, dim=1) # Shape: [subset_size, batch_size_control, 1]

                weighted_values = reshaped_values * probabilities  # Shape: [subset_size, batch_size_control, 1]
                integral = torch.sum(weighted_values, dim=1)  # Shape: [subset_size, 1]
                
                loss = 1e-2*torch.mean((current_value - integral)**2)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sve_net.parameters(), max_norm=10.0)                
                optimizer.step()



    # ==== Auxillary
                
    def clear_episode_memory(self):
        """
        Clears the state-action memory after an episode.
        """
        self.stateaction_memory.clear()
                

    def compute_all_gradient_norms(self) -> List[float]:
        """
        Computes and prints the gradient norms for all State Value Estimators.

        Returns:
            List[float]: List of gradient norms for each SVE network.
        """
        sve_grad_norms = {f"SVE_{i}": round(compute_gradient_norm(sve_net), 5)
                          for i, sve_net in enumerate(self.SVE_networks)}

        return sve_grad_norms


    def save_model(self, path: str):
        """
        Saves the Player instance, excluding non-pickleable environment.

        Args:
            path (str): File path to save the model.
        """
        # Temporarily remove env
        env_backup = self.env
        self.env = None

        try:
            torch.save(self, path)
            print(f"Player instance saved successfully to {path}")
        finally:
            # Restore the env
            self.env = env_backup

