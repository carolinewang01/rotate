import os
import numpy as np
import matplotlib.pyplot as plt

from paper_vis.process_data import load_results_for_task
from paper_vis.plot_globals import TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE

from paper_vis.plot_globals import BASELINES, OUR_METHOD, GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME


def plot_radar_chart(results, metric_name: str, aggregate_stat_name: str,
                   plot_title: str, save: bool, savedir: str, show_plot: bool, savename: str):
    '''Plots a radar chart for a single task, metric, and aggregate stat.'''
    pass


if __name__ == "__main__":
    from paper_vis.plot_globals import BASELINES, OUR_METHOD, GLOBAL_HELDOUT_CONFIG, TASK_TO_PLOT_TITLE, TASK_TO_METRIC_NAME
    
    RESULTS_TO_PLOT = {
        **BASELINES,
        **OUR_METHOD
    }

    task_list = [
        "lbf", 
        "overcooked-v1/asymm_advantages"
    ]