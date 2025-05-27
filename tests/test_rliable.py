from rliable import metrics as rli_metrics
from rliable import library as rli_library
import numpy as np
import matplotlib.pyplot as plt
import scipy

# shape (num_seeds, num_heldout_agents, num_eval_episodes)
num_tasks = 2
data = np.random.uniform(0, 1, (3, num_tasks, 32)) * 10

# First, reshape the data to combine seeds and eval episodes
# We want to aggregate across both, so we'll combine them into a single dimension
# New shape: (num_seeds * num_eval_episodes, num_heldout_agents)
reshaped_data = data.transpose(0, 2, 1).reshape(-1, num_tasks)
# Compute overall IQM score
mean_scores = rli_metrics.aggregate_mean(reshaped_data) # data in form (num_runs, num_tasks)
iqm_scores = rli_metrics.aggregate_iqm(reshaped_data)
print("Overall mean score:", mean_scores)
print("Overall IQM score:", iqm_scores)

iqm_scipy_pertask = scipy.stats.trim_mean(reshaped_data, proportiontocut=0.25, axis=0)
print("IQM score per task (scipy):", iqm_scipy_pertask)

for i in range(num_tasks):
    # Compute bootstrapped CIs
    # We'll use 1000 bootstrap samples and 95% confidence interval
    key = f"heldout_{i}"
    point_estimates, interval_estimates = rli_library.get_interval_estimates(
        {key: reshaped_data[:, [i]]},
        func=lambda x: np.array([rli_metrics.aggregate_iqm(x)]),
        reps=25000,
        confidence_interval_size=0.95
    )

    print(f"\nIQM scores for {key}:", point_estimates[key].shape) # shape (1,)
    print("95% Confidence Intervals:", interval_estimates[key].shape) # shape (2, 1)

    # # Plot the results

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(
    #     range(len(iqm_scores)),
    #     iqm_scores,
    #     yerr=[iqm_scores - ci_low, ci_high - iqm_scores],
    #     fmt='o',
    #     capsize=5
    # )
    # plt.xlabel('Heldout Agent')
    # plt.ylabel('IQM Score')
    # plt.title('IQM Scores with 95% Confidence Intervals')
    # plt.grid(True)
    # plt.show()
