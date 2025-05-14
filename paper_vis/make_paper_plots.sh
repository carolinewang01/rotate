#!/bin/bash

# bar charts for core methods, ablations, and supplemental tasks
python -m paper_vis.bar_charts --plot_type core
python -m paper_vis.bar_charts --plot_type ablations
python -m paper_vis.bar_charts --plot_type supplemental

# radar charts for core methods only
python -m paper_vis.radar_charts --plot_type core

# heldout curves for core tasks and ablations only
python -m paper_vis.heldout_curves --plot_type core
python -m paper_vis.heldout_curves --plot_type ablations
