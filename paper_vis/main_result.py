import os
import numpy as np
import matplotlib.pyplot as plt

from paper_vis.process_data import load_results_for_task
from paper_vis.plot_globals import TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE

plt.rcParams['xtick.labelsize'] = AXIS_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = AXIS_LABEL_FONTSIZE

def plot_single_bar_chart(results, metric_name: str, aggregate_stat_name: str,
                   plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a bar chart for a single task, metric, and aggregate stat.'''
    method_display_names = []
    y_values = []
    y_errors = []

    for display_name, method_results in results.items():
        method_display_names.append(display_name)

        stat_key = f"overall_{aggregate_stat_name}"
        point_estimate = method_results[metric_name][stat_key]
        lower_ci = method_results[metric_name]["overall_lower_ci"]
        upper_ci = method_results[metric_name]["overall_upper_ci"]

        y_values.append(point_estimate)
        # y_errors should be in the format [[lower_errors], [upper_errors]]
        y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])

    num_methods = len(method_display_names)
    x_pos = np.arange(num_methods)

    fig, ax = plt.subplots()
    # Transpose y_errors to match expected format for yerr
    y_errors_transposed = np.array(y_errors).T
    ax.bar(x_pos, y_values, yerr=y_errors_transposed, align='center', alpha=0.7, ecolor='black', capsize=10)

    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} {metric_name.replace("_", " ").title()} (Normalized)', 
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_display_names, rotation=0, ha="center", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    ax.yaxis.grid(True)
    
    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color='dimgray', linestyle='-')

    plt.tight_layout()
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"))
    if show_plot:
        plt.show()

def plot_all_tasks_bar_chart(all_task_results, metric_name: str, aggregate_stat_name: str,
                   plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a bar chart where all tasks are plotted as groups on the same bar chart.'''
    tasks = list(all_task_results.keys())
    num_tasks = len(tasks)
    
    # Get method names from the first task (assuming all tasks have the same methods)
    first_task = tasks[0]
    method_display_names = list(all_task_results[first_task].keys())
    num_methods = len(method_display_names)
    
    # Width of each bar
    bar_width = 0.8 / num_methods
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(max(8, num_tasks * 2), 6))
    
    # Set up x positions for tasks and bars within each task group
    task_positions = np.arange(num_tasks)
    
    # Colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, num_methods*2))
    
    # Plot bars for each method across all tasks
    for i, method_name in enumerate(method_display_names):
        y_values = []
        y_errors = []
        
        for task in tasks:
            method_results = all_task_results[task][method_name]
            task_metric_name = TASK_TO_METRIC_NAME[task] if metric_name == "task_specific" else metric_name
            
            stat_key = f"overall_{aggregate_stat_name}"
            point_estimate = method_results[task_metric_name][stat_key]
            lower_ci = method_results[task_metric_name]["overall_lower_ci"]
            upper_ci = method_results[task_metric_name]["overall_upper_ci"]
            
            y_values.append(point_estimate)
            y_errors.append([point_estimate - lower_ci, upper_ci - point_estimate])
        
        # Position bars for this method within each task group
        x_positions = task_positions + (i - num_methods/2 + 0.5) * bar_width
        
        # Transpose y_errors to match expected format for yerr
        y_errors_transposed = np.array(y_errors).T
        
        # Plot bars for this method
        ax.bar(x_positions, y_values, width=bar_width, label=method_name, 
               yerr=y_errors_transposed, alpha=0.7, color=colors[i], 
               ecolor='black', capsize=5)
    
    # Set x-axis tick labels to task names
    task_display_names = [task.replace("overcooked-v1/", "") for task in tasks]
    ax.set_xticks(task_positions)
    ax.set_xticklabels(task_display_names, rotation=25, ha="right", fontsize=AXIS_LABEL_FONTSIZE)

    # Set labels and title
    ax.set_ylabel(f'{aggregate_stat_name.capitalize()} {metric_name.replace("_", " ").title() if metric_name != "task_specific" else "Performance"} (Normalized)', 
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE)
    
    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color='dimgray', linestyle='-')
    
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{savename}.pdf"))
    if show_plot:
        plt.show()


if __name__ == "__main__":
    from paper_vis.plot_globals import BASELINES, OUR_METHOD, ABLATIONS, SUPPLEMENTAL, \
        GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME
    
    PLOT_TYPE = "supplemental" # core or ablations or supplemental
    if PLOT_TYPE == "core":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **BASELINES 
        }
    elif PLOT_TYPE == "ablations":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **ABLATIONS
        }
    elif PLOT_TYPE == "supplemental":
        RESULTS_TO_PLOT = {
            **OUR_METHOD,
            **SUPPLEMENTAL
        }

    task_list = [
        # "lbf", 
        "overcooked-v1/cramped_room",
        "overcooked-v1/asymm_advantages",
        "overcooked-v1/counter_circuit",
        "overcooked-v1/coord_ring",
        "overcooked-v1/forced_coord"

                 ]
    all_task_results = {}
    for task_name in task_list:
        all_task_results[task_name] = load_results_for_task(task_name, RESULTS_TO_PLOT, load_from_cache=True)
        metric_name = TASK_TO_METRIC_NAME[task_name]

        # Individual task plots
        PLOT_ARGS = {
            "save": True,
            "savedir": "results/neurips_figures", 
            "savename": f"{task_name.replace('/', '_')}_{PLOT_TYPE}_bar",
            "plot_title": TASK_TO_PLOT_TITLE[task_name],
            "show_plot": False
        }
        
        # plot_single_bar_chart(all_task_results[task_name], 
        #             metric_name=metric_name, 
        #             aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
        #             **PLOT_ARGS
        #             )
    
    # Plot all tasks together
    ALL_TASKS_PLOT_ARGS = {
        "save": True,
        "savedir": "results/neurips_figures", 
        "savename": f"all_tasks_comparison_{PLOT_TYPE}",
        "plot_title": "Method Performance Across All Tasks",
        "show_plot": True
    }
    
    plot_all_tasks_bar_chart(all_task_results, 
                metric_name="task_specific", 
                aggregate_stat_name=GLOBAL_HELDOUT_CONFIG["global_heldout_settings"]["AGGREGATE_STAT"],
                **ALL_TASKS_PLOT_ARGS
                )
