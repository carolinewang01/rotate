#!/usr/bin/env python
"""
Script to compute, save, and use the best observed returns for heldout agents.

This module provides functionality to:
1. Compute and save the best unnormalized returns across all methods for each heldout agent
2. Use these best returns for normalizing metrics for visualization

The normalization process works as follows:
- Original metrics are saved in a normalized form using predefined performance bounds
- Each heldout agent has its own specific performance bounds [lower_bound, upper_bound]
- This script first unnormalizes the metrics using each agent's original bounds: 
  unnormalized = normalized * (upper_bound - lower_bound) + lower_bound
- Then we find the best unnormalized return across all methods for each heldout agent
- These best returns are saved and used to normalize metrics for visualization: 
  new_normalized = unnormalized / best_return

The best returns for each agent are always guaranteed to be at least as high as that
agent's original upper bound from the performance bounds, ensuring we don't make
things look worse than the original normalization.
"""

import os
import numpy as np
import pickle
import json
from typing import Dict, List

from common.save_load_utils import load_train_run
from common.plot_utils import get_metric_names
from paper_vis.plot_globals import TASK_TO_ENV_NAME, get_heldout_agents

def get_original_performance_bounds(task_name: str) -> Dict[str, List[List[float]]]:
    """Get the original agent-specific performance bounds for a task.
    
    Returns:
        Dict mapping metric names to a list of [lower_bound, upper_bound] pairs for each agent
    """
    task_config_path = f"open_ended_training/configs/task/{task_name.replace('-v1', '')}.yaml"
    heldout_agents_dict = get_heldout_agents(task_name, task_config_path)
    
    # Extract the agent-specific performance bounds
    agent_bounds = []
    for agent_name, agent_dict in heldout_agents_dict.items():
        bounds = agent_dict[3]  # The performance bounds are at index 3
        agent_bounds.append((agent_name, bounds))
        
    # Reorganize by metric
    metric_bounds = {}
    for _, bounds in agent_bounds:
        for metric_name, bound_values in bounds.items():
            if metric_name not in metric_bounds:
                metric_bounds[metric_name] = []
            metric_bounds[metric_name].append(bound_values)
    
    return metric_bounds


def extract_returns_for_method(eval_metrics_dir: str, env_name: str, is_oel_method: bool, task_name: str) -> Dict[str, np.ndarray]:
    """Extract the returns for each heldout agent from a method's evaluation metrics.
    
    The metrics are first unnormalized using the original performance bounds
    before calculating means and returning them. Each heldout agent has its own
    specific performance bounds.
    """
    print(f"Processing {eval_metrics_dir} (OEL method: {is_oel_method})")
    
    try:
        # Load the normalized evaluation metrics
        eval_metrics = load_train_run(eval_metrics_dir)
        metric_names = get_metric_names(env_name)
        
        if not metric_names:
            raise ValueError(f"No metric names found for environment {env_name}")
        
        # Get the original performance bounds for each heldout agent
        task_config_path = f"open_ended_training/configs/task/{task_name.replace('-v1', '')}.yaml"
        heldout_agents_dict = get_heldout_agents(task_name, task_config_path)
        
        # Extract the agent-specific performance bounds
        agent_performance_bounds = []
        for agent_name, agent_dict in heldout_agents_dict.items():
            bounds = agent_dict[3]  # The performance bounds are at index 3
            agent_performance_bounds.append((agent_name, bounds))
        
        print(f"Found performance bounds for {len(agent_performance_bounds)} heldout agents")
        
        returns_per_heldout_agent = {}
        
        for metric_name in metric_names:
            if metric_name not in eval_metrics:
                print(f"Warning: Metric {metric_name} not found in evaluation metrics")
                continue
            
            try:
                data = eval_metrics[metric_name]
                # Create a copy for unnormalization
                unnormalized_data = np.copy(data)
                if unnormalized_data.max() > 10: 
                    raise ValueError(f"Unnormalized data at {eval_metrics_dir} has values greater than 10.")
                
                if is_oel_method:
                    if len(data.shape) != 5:
                        print(f"Warning: Expected OEL data shape of 5 dimensions but got {len(data.shape)} dimensions")
                        continue
                    
                    # For OEL data, the heldout agent dimension is at index 2
                    num_heldout_agents = data.shape[2]
                    
                    # Check that we have the right number of bounds
                    if num_heldout_agents != len(agent_performance_bounds):
                        print(f"Warning: Number of heldout agents in data ({num_heldout_agents}) "
                              f"doesn't match number of agents with bounds ({len(agent_performance_bounds)})")
                    
                    # Unnormalize each heldout agent using its specific bounds
                    for h in range(min(num_heldout_agents, len(agent_performance_bounds))):
                        agent_name, bounds = agent_performance_bounds[h]
                        if metric_name in bounds:
                            lower_bd, upper_bd = bounds[metric_name]
                            print(f"Unnormalizing agent {h} ({agent_name}) using bounds: [{lower_bd}, {upper_bd}]")
                            
                            # The stored metrics are normalized to around 0-1 using:
                            # (value - lower) / (upper - lower)
                            # We need to unnormalize them correctly using:
                            # value * (upper - lower) + lower
                            unnormalized_data[:, :, h, :, :] = data[:, :, h, :, :] * (upper_bd - lower_bd) + lower_bd
                    
                    # Take the mean over agents per game, then mean over episodes
                    mean_data = unnormalized_data.mean(axis=-1)  # Now shape (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes)
                    # For best returns, we want the mean performance over seeds, iterations, and episodes
                    mean_returns = mean_data.mean(axis=-1).max(axis=(0, 1))  # Shape (num_heldout_agents,)
                
                elif len(data.shape) == 4:  # Teammate generation methods
                    # For teammate gen data, the heldout agent dimension is at index 1
                    num_heldout_agents = data.shape[1]
                    
                    # Check that we have the right number of bounds
                    if num_heldout_agents != len(agent_performance_bounds):
                        print(f"Warning: Number of heldout agents in data ({num_heldout_agents}) "
                              f"doesn't match number of agents with bounds ({len(agent_performance_bounds)})")
                    
                    # Unnormalize each heldout agent using its specific bounds
                    for h in range(min(num_heldout_agents, len(agent_performance_bounds))):
                        agent_name, bounds = agent_performance_bounds[h]
                        if metric_name in bounds:
                            lower_bd, upper_bd = bounds[metric_name]
                            print(f"Unnormalizing agent {h} ({agent_name}) using bounds: [{lower_bd}, {upper_bd}]")
                            
                            # The stored metrics are already normalized to [0,1] using:
                            # (value - lower) / (upper - lower)
                            # We need to unnormalize them correctly using:
                            # value * (upper - lower) + lower
                            unnormalized_data[:, h, :, :] = data[:, h, :, :] * (upper_bd - lower_bd) + lower_bd
                    
                    # Take the mean over agents per game, then mean over episodes and seeds
                    mean_data = unnormalized_data.mean(axis=-1)  # Now shape (num_seeds, num_heldout_agents, num_eval_episodes)
                    mean_returns = mean_data.mean(axis=-1).max(axis=0)  # Shape (num_heldout_agents,)                
                else:
                    print(f"Warning: Unexpected data shape {data.shape} for {metric_name}")
                    continue
                
                # Sanity check for percentage metrics
                if metric_name == "percent_eaten" and np.any(mean_returns > 100):
                    raise ValueError(f"Warning: unnormalized percent_eaten values exceed 100%. Max value: {mean_returns.max()}")
                
                # Sanity check for Overcooked returns
                if "overcooked" in task_name and (metric_name == "returned_episode_returns" or metric_name == "base_return"):
                    if np.any(mean_returns > 500):
                        raise ValueError(f"Warning: unnormalized Overcooked returns are unusually high. Max value: {mean_returns.max()}")
                
                print(f"Mean returns shape for {metric_name}: {mean_returns.shape}, values: {mean_returns}")
                returns_per_heldout_agent[metric_name] = mean_returns
            
            except Exception as e:
                print(f"Error processing metric {metric_name}: {e}")
        
        return returns_per_heldout_agent
    
    except Exception as e:
        print(f"Error loading from {eval_metrics_dir}: {e}")
        return {}


def find_eval_metrics_dirs(base_dir: str) -> List[Dict]:
    """Find all heldout_eval_metrics directories recursively and determine if they're from OEL methods.
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        List of dicts with structure {'dir': directory_path, 'is_oel': boolean}
    """
    eval_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        if "heldout_eval_metrics" in dirs:
            # Found a heldout_eval_metrics directory
            eval_metrics_dir = os.path.join(root, "heldout_eval_metrics")
            
            # Determine if this is from an OEL method based on the path
            is_oel_method = "open_ended" in root or "oe_" in root
            
            eval_dirs.append({
                'dir': eval_metrics_dir,
                'is_oel': is_oel_method
            })
    
    return eval_dirs


def compute_best_returns(task_name: str) -> Dict[str, List[float]]:
    """Compute the best observed returns for each heldout agent across all methods.
    
    This function loads the evaluation metrics, unnormalizes them using the original
    agent-specific performance bounds, and then finds the best returns across all 
    methods for each heldout agent.
    
    The best returns are guaranteed to be at least as high as the original agent-specific
    upper bounds from the performance bounds.
    
    Returns:
        Dict with keys:
        - metric_name: List of best returns for each heldout agent
        - metric_name_method: List of method paths that achieved each best return
        - metric_name_seed: List of seed indices that achieved each best return
        - metric_name_iter: List of iteration indices that achieved each best return
    """
    task_dir = f"results/{task_name}"
    env_name = TASK_TO_ENV_NAME[task_name]
    
    # Find all evaluation metrics directories for this task
    print(f"Searching for evaluation metrics in {task_dir}...")
    eval_metrics_dirs = find_eval_metrics_dirs(task_dir)
    
    if not eval_metrics_dirs:
        raise ValueError(f"No evaluation metrics found for task {task_name}")
    
    print(f"Found {len(eval_metrics_dirs)} evaluation directories.")
    
    # Get original performance bounds to ensure our best returns are at least as high
    original_bounds = get_original_performance_bounds(task_name)
    print(f"Original performance bounds: {original_bounds}")
    
    # First, examine a method's eval metrics to determine the number of heldout agents
    num_heldout_agents = None
    for eval_dir_info in eval_metrics_dirs:
        eval_metrics_dir = eval_dir_info['dir']
        try:
            eval_metrics = load_train_run(eval_metrics_dir)
            metric_names = get_metric_names(env_name)
            # Find the number of heldout agents from the data shape
            for metric_name in metric_names:
                if metric_name not in eval_metrics:
                    continue
                    
                metric_data = eval_metrics[metric_name]
                if len(metric_data.shape) == 5:  # OEL method
                    num_heldout_agents = metric_data.shape[2]
                elif len(metric_data.shape) == 4:  # Teammate generation method
                    num_heldout_agents = metric_data.shape[1]
                
                if num_heldout_agents is not None:
                    print(f"Detected {num_heldout_agents} heldout agents from {eval_metrics_dir}")
                    break
            
            if num_heldout_agents is not None:
                break
        except Exception as e:
            print(f"Error examining {eval_metrics_dir}: {e}")
    
    if num_heldout_agents is None:
        # Default to 6 heldout agents if we couldn't detect it
        num_heldout_agents = 6
        print(f"Warning: Could not determine number of heldout agents for {task_name}. Using default value: {num_heldout_agents}")
    
    # Initialize dicts to store best returns and their metadata for each heldout agent and each metric
    best_returns = {}
    best_returns_method = {}
    best_returns_seed = {}
    best_returns_iter = {}
    
    # Process each evaluation directory to find best returns
    for eval_dir_info in eval_metrics_dirs:
        eval_metrics_dir = eval_dir_info['dir']
        is_oel_method = eval_dir_info['is_oel']
        
        try:
            # Extract returns from this method (these are already unnormalized)
            returns = extract_returns_for_method(eval_metrics_dir, env_name, is_oel_method, task_name)
            
            # Update best returns
            for metric_name, values in returns.items():
                if metric_name not in best_returns:
                    # First time seeing this metric, initialize the arrays
                    best_returns[metric_name] = values
                    best_returns_method[metric_name] = [eval_metrics_dir] * len(values)
                    best_returns_seed[metric_name] = [0] * len(values)  # Default seed
                    best_returns_iter[metric_name] = [0] * len(values)  # Default iteration
                else:
                    # Make sure arrays are the same length
                    if len(values) != len(best_returns[metric_name]):
                        print(f"Warning: Skipping {eval_metrics_dir} for metric {metric_name} - "
                              f"expected {len(best_returns[metric_name])} values but got {len(values)}")
                        continue
                    
                    # Compare and update best returns and their metadata
                    for i in range(len(values)):
                        if values[i] > best_returns[metric_name][i]:
                            best_returns[metric_name][i] = values[i]
                            best_returns_method[metric_name][i] = eval_metrics_dir
                            # For OEL methods, we need to find which seed and iteration achieved this return
                            if is_oel_method:
                                eval_metrics = load_train_run(eval_metrics_dir)
                                metric_data = eval_metrics[metric_name]
                                # Find the seed and iteration that achieved this return
                                best_seed, best_iter = np.unravel_index(
                                    np.argmax(metric_data[:, :, i].mean(axis=-1)), 
                                    metric_data[:, :, i].mean(axis=-1).shape
                                )
                                best_returns_seed[metric_name][i] = int(best_seed)
                                best_returns_iter[metric_name][i] = int(best_iter)
                            else:
                                # For teammate generation methods, find the seed that achieved this return
                                eval_metrics = load_train_run(eval_metrics_dir)
                                metric_data = eval_metrics[metric_name]
                                best_seed = np.argmax(metric_data[:, i].mean(axis=-1))
                                best_returns_seed[metric_name][i] = int(best_seed)
                                best_returns_iter[metric_name][i] = 0  # No iterations for teammate gen methods
        except Exception as e:
            print(f"Error processing {eval_metrics_dir}: {e}")
    
    # Check if we found any valid returns
    if not best_returns:
        raise ValueError(f"No valid returns found for any metric in task {task_name}")
    
    # Ensure best returns are at least as high as original agent-specific upper bounds
    for metric_name, values in best_returns.items():
        if metric_name in original_bounds:
            agent_bounds = original_bounds[metric_name]
            print(f"For metric {metric_name}, original bounds: {agent_bounds}")
            print(f"Current best returns: {values}")
            
            # Ensure all values are at least as high as the corresponding agent's original upper bound
            for i in range(min(len(values), len(agent_bounds))):
                original_upper_bound = agent_bounds[i][1]  # Index 1 is the upper bound
                if values[i] < original_upper_bound:
                    print(f"Adjusting best return for heldout agent {i} from {values[i]} to {original_upper_bound}")
                    values[i] = original_upper_bound
    
    # Convert numpy arrays to lists for JSON serialization
    for metric_name in best_returns:
        best_returns[metric_name] = best_returns[metric_name].tolist()
        best_returns_method[metric_name] = best_returns_method[metric_name]
        best_returns_seed[metric_name] = best_returns_seed[metric_name]
        best_returns_iter[metric_name] = best_returns_iter[metric_name]
    
    # Combine all the information into a single dictionary
    result = {}
    for metric_name in best_returns:
        result[metric_name] = best_returns[metric_name]
        result[f"{metric_name}_method"] = best_returns_method[metric_name]
        result[f"{metric_name}_seed"] = best_returns_seed[metric_name]
        result[f"{metric_name}_iter"] = best_returns_iter[metric_name]
    
    return result


def save_best_returns(task_name: str, best_returns: Dict[str, List[float]]):
    """Save the computed best returns to a file."""
    results_dir = f"results/{task_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "best_heldout_returns.json")
    with open(output_file, "w") as f:
        json.dump(best_returns, f, indent=2)
    
    print(f"Saved best returns to {output_file}")


def load_best_returns(task_name: str) -> Dict[str, List[float]]:
    """Load the best returns for a task."""
    file_path = f"results/{task_name}/best_heldout_returns.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        print(f"Best returns file not found for {task_name}. Computing...")
        best_returns = compute_best_returns(task_name)
        save_best_returns(task_name, best_returns)
        return best_returns


def unnormalize_data(eval_metrics: Dict[str, np.ndarray], config, task_name: str) -> Dict[str, np.ndarray]:
    '''Unnormalize the data based on the original performance bounds for each metric.'''
    # Get original performance bounds from the heldout agents config
    task_config_path = f"open_ended_training/configs/task/{task_name.replace('-v1', '')}.yaml"
    heldout_agents_dict = get_heldout_agents(task_name, task_config_path)
    
    # Extract performance bounds (same bounds apply to all heldout agents)
    performance_bounds = {}
    for _, agent_dict in heldout_agents_dict.items():
        bounds = agent_dict[3]  # The performance bounds are at index 3
        for metric_name, bound_values in bounds.items():
            if metric_name not in performance_bounds:
                performance_bounds[metric_name] = bound_values
    
    # Unnormalize the eval metrics by reversing the min-max normalization
    unnormalized_metrics = {}
    for metric_name, data in eval_metrics.items():
        if metric_name in performance_bounds:
            lower_bd, upper_bd = performance_bounds[metric_name]
            # Reverse the min-max normalization: original = normalized * (upper - lower) + lower
            unnormalized_metrics[metric_name] = data * (upper_bd - lower_bd) + lower_bd
        else:
            # If we don't have bounds for this metric, keep it as is
            unnormalized_metrics[metric_name] = data
    
    return unnormalized_metrics


def renormalize_eval_metrics(eval_metrics: Dict[str, np.ndarray], 
                           task_name: str, 
                           best_returns: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """Renormalize evaluation metrics using the best observed returns.
    
    This function takes the raw (normalized) metrics, unnormalizes them using
    agent-specific performance bounds, and then normalizes them using the best
    observed returns: new_normalized = unnormalized / best_return
    """
    # First get the agent-specific performance bounds
    task_config_path = f"open_ended_training/configs/task/{task_name.replace('-v1', '')}.yaml"
    heldout_agents_dict = get_heldout_agents(task_name, task_config_path)
    
    # Extract the agent-specific performance bounds
    agent_performance_bounds = []
    for agent_name, agent_dict in heldout_agents_dict.items():
        bounds = agent_dict[3]  # The performance bounds are at index 3
        agent_performance_bounds.append((agent_name, bounds))
    
    
    # Now unnormalize and then renormalize each metric
    normalized_metrics = {}
    for metric_name, data in eval_metrics.items():
        # Create a copy to avoid modifying the original data
        unnormalized_data = np.copy(data)
        shape = data.shape
        
        # Determine the number of heldout agents and their dimension index
        if len(shape) == 5:  # OEL methods: (num_seeds, num_oel_iter, num_heldout_agents, num_eval_episodes, num_agents_per_game)
            num_heldout_agents = shape[2]
            heldout_dim = 2
        elif len(shape) == 4:  # Teammate generation methods: (num_seeds, num_heldout_agents, num_eval_episodes, num_agents_per_game)
            num_heldout_agents = shape[1]
            heldout_dim = 1
        else:
            print(f"Warning: Unexpected shape {shape} for {metric_name}. Keeping original data.")
            normalized_metrics[metric_name] = data
            continue
        
        # Check that we have the right number of bounds and best returns
        if num_heldout_agents != len(agent_performance_bounds):
            print(f"Warning: Number of heldout agents in data ({num_heldout_agents}) "
                 f"doesn't match number of agents with bounds ({len(agent_performance_bounds)})")

        if metric_name not in best_returns:
            # print(f"Warning: No best returns found for metric {metric_name}. Keeping original data.")
            normalized_metrics[metric_name] = data
            continue
        
        if num_heldout_agents != len(best_returns[metric_name]):
            print(f"Warning: Number of heldout agents in data ({num_heldout_agents}) "
                 f"doesn't match number of best returns ({len(best_returns[metric_name])})")
        
        # Determine the number of agents to process
        agents_to_process = min(num_heldout_agents, len(agent_performance_bounds), len(best_returns[metric_name]))
        
        # Unnormalize each heldout agent using its specific bounds, then renormalize using best returns
        for h in range(agents_to_process):
            agent_name, bounds = agent_performance_bounds[h]
            if metric_name in bounds:
                # Get the original bounds for this agent and metric
                lower_bd, upper_bd = bounds[metric_name]
                # Get the best return for this agent and metric
                best_return = best_returns[metric_name][h]
                
                if heldout_dim == 2:  # OEL methods
                    # Unnormalize: original = normalized * (upper - lower) + lower
                    temp_unnormalized = data[:, :, h, :, :] * (upper_bd - lower_bd) + lower_bd
                    # Renormalize using best return
                    if best_return > 0:
                        unnormalized_data[:, :, h, :, :] = temp_unnormalized / best_return
                    else:
                        print(f"Warning: Best return for metric {metric_name}, agent {h} is {best_return}. Cannot normalize.")
                        unnormalized_data[:, :, h, :, :] = data[:, :, h, :, :]  # Keep original
                
                elif heldout_dim == 1:  # Teammate generation methods
                    # Unnormalize: original = normalized * (upper - lower) + lower
                    temp_unnormalized = data[:, h, :, :] * (upper_bd - lower_bd) + lower_bd
                    # Renormalize using best return
                    if best_return > 0:
                        unnormalized_data[:, h, :, :] = temp_unnormalized / best_return
                    else:
                        print(f"Warning: Best return for metric {metric_name}, agent {h} is {best_return}. Cannot normalize.")
                        unnormalized_data[:, h, :, :] = data[:, h, :, :]  # Keep original
        
        normalized_metrics[metric_name] = unnormalized_data
    
    return normalized_metrics


def find_all_tasks():
    """Find all available tasks in the results directory."""
    tasks = []
    results_dir = "results"
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item in TASK_TO_ENV_NAME:
                tasks.append(item)
            elif os.path.isdir(item_path) and item == "overcooked-v1":
                # Handle overcooked tasks which have subdirectories
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        task = f"overcooked-v1/{subitem}"
                        if task in TASK_TO_ENV_NAME:
                            tasks.append(task)
    return tasks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute, save and use best observed returns for heldout agents")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Command to compute best returns
    compute_parser = subparsers.add_parser("compute", help="Compute and save best observed returns")
    compute_parser.add_argument("--tasks", nargs="+", 
                               help="List of tasks to process (e.g., lbf overcooked-v1/cramped_room). If not provided, all tasks will be processed.")
    compute_parser.add_argument("--force", action="store_true", 
                               help="Force recomputation even if best returns file exists")
    
    # Command to list available tasks
    list_parser = subparsers.add_parser("list", help="List all available tasks")
    
    # Command to show best returns for a task
    show_parser = subparsers.add_parser("show", help="Show best returns for a task")
    show_parser.add_argument("--tasks", nargs="+", 
                            help="List of tasks to show best returns for. If not provided, all tasks will be shown.")
    
    args = parser.parse_args()
    
    if args.command == "compute":
        # Get task list
        if not args.tasks:
            tasks = find_all_tasks()
            if not tasks:
                print("No tasks found in results directory. Please specify tasks manually.")
                exit(1)
        else:
            tasks = args.tasks
        
        print(f"Processing {len(tasks)} tasks: {tasks}")
        
        # Process each task
        for task_name in tasks:
            print(f"\nProcessing task: {task_name}")
            
            # Check if best returns file already exists
            best_returns_file = f"results/{task_name}/best_heldout_returns.json"
            if os.path.exists(best_returns_file) and not args.force:
                print(f"Best returns file already exists for {task_name}. Use --force to recompute.")
                continue
            
            # Compute and save best returns
            try:
                best_returns = compute_best_returns(task_name)
                save_best_returns(task_name, best_returns)
                print(f"Successfully computed and saved best returns for {task_name}")
                
                # Print summary
                for metric_name, values in best_returns.items():
                    print(f"  {metric_name}: {values}")
            except Exception as e:
                print(f"Error processing {task_name}: {e}")
    
    elif args.command == "list":
        tasks = find_all_tasks()
        if tasks:
            print("Available tasks:")
            for task in sorted(tasks):
                print(f"  {task}")
        else:
            print("No tasks found in results directory.")
    
    elif args.command == "show":
        if not args.tasks:
            tasks = find_all_tasks()
        else:
            tasks = args.tasks
        
        for task_name in tasks:
            print(f"Best returns for {task_name}:")
            try:
                best_returns = load_best_returns(task_name)
                for metric_name, values in best_returns.items():
                    print(f"  {metric_name}: {values}")
            except Exception as e:
                print(f"  Error loading best returns: {e}")
    
    else:
        parser.print_help() 