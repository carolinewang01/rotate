#!/bin/bash

# List all available tasks
python -m paper_vis.compute_best_returns list

# Compute best returns for all available tasks
python -m paper_vis.compute_best_returns compute

# Compute best returns for specific tasks
python -m paper_vis.compute_best_returns compute --tasks lbf overcooked-v1/cramped_room

# Force recomputation of best returns
python -m paper_vis.compute_best_returns compute --tasks lbf --force

# Show the computed best returns for specific tasks
python -m paper_vis.compute_best_returns show --tasks lbf overcooked-v1/cramped_room

# Generate visualizations using the best returns normalization
python -m paper_vis.bar_charts --plot_type core --show_plots

# Generate visualizations using the original normalization
python -m paper_vis.bar_charts --plot_type core --use_original_normalization --show_plots