import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from vis.process_data import load_results_for_task
from vis.plot_globals import TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TASK_TO_AXIS_DISPLAY_NAME

plt.rcParams['xtick.labelsize'] = AXIS_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = AXIS_LABEL_FONTSIZE


def generate_table(all_task_results, metric_name: str, aggregate_stat_name: str,
                   save: bool, savedir: str, savename: str):
    '''Generates a table where each column represents a task and each row is a method's results.'''
    
    # Get all tasks and methods
    tasks = list(all_task_results.keys())
    
    # Get method names from the first task (assuming all tasks have the same methods)
    first_task = tasks[0]
    method_display_names = list(all_task_results[first_task].keys())
    
    # Initialize the table data
    table_data = {}
    
    # Process each task
    for task in tasks:
        task_display_name = TASK_TO_AXIS_DISPLAY_NAME[task]
        table_data[task_display_name] = []
        
        for method_name in method_display_names:
            method_results = all_task_results[task][method_name]
            
            # Determine the metric name for this task
            from vis.plot_globals import TASK_TO_METRIC_NAME
            task_metric_name = TASK_TO_METRIC_NAME[task] if metric_name == "task_specific" else metric_name
            
            # Extract the statistics
            stat_key = f"overall_{aggregate_stat_name}"
            point_estimate = method_results[task_metric_name][stat_key]
            lower_ci = method_results[task_metric_name]["overall_lower_ci"]
            upper_ci = method_results[task_metric_name]["overall_upper_ci"]
            
            # Format as "return (lower CI, upper CI)" with appropriate precision
            formatted_cell = f"{point_estimate:.3f} ({lower_ci:.3f}, {upper_ci:.3f})"
            table_data[task_display_name].append(formatted_cell)
    
    # Create DataFrame with methods as index
    df = pd.DataFrame(table_data, index=method_display_names)
    
    # Display the table
    print("\nResults Table:")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    
    # Save the table if requested
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        # Save as CSV
        csv_path = os.path.join(savedir, f"{savename}.csv")
        df.to_csv(csv_path)
        print(f"Saved table to {csv_path}")
        
        # Save as markdown file
        md_path = os.path.join(savedir, f"{savename}.md")
        with open(md_path, 'w') as f:
            f.write("# Results Table\n\n")
            f.write(df.to_markdown())
            f.write("\n")
        print(f"Saved markdown table to {md_path}")


if __name__ == "__main__":
    from vis.plot_globals import OE_BASELINES, TEAMMATE_GEN_BASELINES, OUR_METHOD, ABLATIONS_OBJ, ABLATIONS_POP, SUPPLEMENTAL, \
        GLOBAL_HELDOUT_CONFIG, TASK_TO_METRIC_NAME, CACHE_FILENAME, RESULTS_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate tables for visualization")
    parser.add_argument("--plot_type", type=str, default="core",
                        choices=["core", "ablations_obj", "ablations_pop", "supplemental"],
                        help="Type of results to include in table")
    parser.add_argument("--use_original_normalization", action="store_true",
                        help="Use original normalization instead of best-returns normalization")
    parser.add_argument("--save_dir", type=str, default=f"{RESULTS_DIR}/tables",
                        help="Directory to save tables")
    parser.add_argument("--tasks", nargs="+", 
                        help="List of tasks to show best returns for. If not provided, all tasks will be computed.")
    args = parser.parse_args()
    
    if args.plot_type == "core":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **OE_BASELINES,
            **TEAMMATE_GEN_BASELINES
        }
    elif args.plot_type == "ablations_obj":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **ABLATIONS_OBJ,
        }
    elif args.plot_type == "ablations_pop":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **ABLATIONS_POP,
        }
    elif args.plot_type == "supplemental":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **SUPPLEMENTAL
        }

    if not args.tasks:
        task_list = [
            "lbf", 
            "overcooked-v1/cramped_room",
            "overcooked-v1/asymm_advantages",
            "overcooked-v1/counter_circuit",
            "overcooked-v1/coord_ring",
            "overcooked-v1/forced_coord"
        ]
    else:
        task_list = args.tasks
    
    # Add suffix to savename based on normalization method
    norm_suffix = "original_normalization" if args.use_original_normalization else "br_normalization"
    
    all_task_results = {}
    for task_name in task_list:
        # Pass the normalization flag to load_results_for_task
        all_task_results[task_name] = load_results_for_task(
            task_name, 
            RESULTS_DIR,
            RESULTS_TO_PLOT, 
            CACHE_FILENAME, 
            load_from_cache=True,
            renormalize_metrics=not args.use_original_normalization
        )
        metric_name = TASK_TO_METRIC_NAME[task_name]
    
    # Generate table for all tasks together
    ALL_TASKS_TABLE_ARGS = {
        "save": True,
        "savedir": args.save_dir, 
        "savename": f"all_tasks_comparison_{args.plot_type}_{norm_suffix}",
    }
    
    generate_table(all_task_results, 
                   metric_name="task_specific", 
                   aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
                   **ALL_TASKS_TABLE_ARGS
                   )
