import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from paper_vis.process_data import load_results_for_task
from paper_vis.plot_globals import TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE

plt.rcParams['xtick.labelsize'] = AXIS_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = AXIS_LABEL_FONTSIZE


def plot_single_task_heldout_curves(results, metric_name: str, aggregate_stat_name: str,
                        plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots learning curves for different methods on a single task, showing how performance changes over iterations.'''
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color='dimgray', linestyle='--')
    
    # Colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)*2))
    
    for i, (display_name, method_results) in enumerate(results.items()):
        # Get the data for this method
        y_values = method_results[metric_name][f"overall_{aggregate_stat_name}"]
        lower_ci = method_results[metric_name]["overall_lower_ci"]
        upper_ci = method_results[metric_name]["overall_upper_ci"]
        
        # Create x-axis values (iterations)
        x_values = np.arange(len(y_values))
        
        # Plot the line with error bands
        ax.plot(x_values, y_values, label=display_name, color=colors[i], linewidth=2)
        ax.fill_between(x_values, lower_ci, upper_ci, color=colors[i], alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('Iteration', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} {metric_name.replace("_", " ").title()} (Normalized)', 
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    
    # Add legend
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"))
    if show_plot:
        plt.show()

if __name__ == "__main__":
    from paper_vis.plot_globals import OE_BASELINES, OUR_METHOD, ABLATIONS, SUPPLEMENTAL, \
        GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME, HELDOUT_CURVES_CACHE_FILENAME
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate heldout learning curves for visualization")
    parser.add_argument("--plot_type", type=str, default="core",
                        choices=["core", "ablations", "supplemental"],
                        help="Type of plot to generate")
    parser.add_argument("--use_original_normalization", action="store_true",
                        help="Use original normalization instead of best-returns normalization")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots in addition to saving them")
    parser.add_argument("--save_dir", type=str, default="results/neurips_figures",
                        help="Directory to save plots")
    parser.add_argument("--tasks", nargs="+", 
                        help="List of tasks to show best returns for. If not provided, all tasks will be computed.")
    args = parser.parse_args()
    
    if args.plot_type == "core":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **OE_BASELINES,
        }
    elif args.plot_type == "ablations":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **ABLATIONS
        }
    elif args.plot_type == "supplemental":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **SUPPLEMENTAL
        }

    # Add suffix to savename based on normalization method
    norm_suffix = "original_normalization" if args.use_original_normalization else "br_normalization"

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
    
    all_task_results = {}
    for task_name in task_list:
        all_task_results[task_name] = load_results_for_task(
            task_name, 
            RESULTS_TO_PLOT, 
            HELDOUT_CURVES_CACHE_FILENAME, 
            load_from_cache=True,
            renormalize_metrics=not args.use_original_normalization
        )
        metric_name = TASK_TO_METRIC_NAME[task_name]

        # Individual task plots
        PLOT_ARGS = {
            "save": True,
            "savedir": f"{args.save_dir}/{task_name}", 
            "savename": f"heldout_curves_{args.plot_type}_{norm_suffix}",
            "plot_title": TASK_TO_PLOT_TITLE[task_name],
            "show_plot": args.show_plots
        }
        plot_single_task_heldout_curves(all_task_results[task_name], metric_name, 
            aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
            **PLOT_ARGS)
        
       