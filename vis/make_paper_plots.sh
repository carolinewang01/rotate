#!/bin/bash

# bar charts for core methods, ablations, and supplemental tasks
python paper_vis/bar_charts.py --plot_type core
python paper_vis/bar_charts.py --plot_type ablations_obj
python paper_vis/bar_charts.py --plot_type ablations_pop
python paper_vis/bar_charts.py --plot_type supplemental

# radar charts for core methods only
python paper_vis/radar_charts.py --plot_type core

# heldout curves for rotate variations
python paper_vis/heldout_curves.py --plot_type all_rotate_vars
