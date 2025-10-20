#!/bin/bash

# bar charts for core methods, ablations, and supplemental tasks
python vis/bar_charts.py --plot_type core
python vis/bar_charts.py --plot_type ablations_obj
python vis/bar_charts.py --plot_type ablations_pop
python vis/bar_charts.py --plot_type supplemental

# radar charts for core methods only
python vis/radar_charts.py --plot_type core

# heldout curves for rotate variations
python vis/heldout_curves.py --plot_type all_rotate_vars
