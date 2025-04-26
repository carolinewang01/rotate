import numpy as np

table_data = np.zeros((2, 3))
metric_names = ("returned_episode_returns", "percent_eaten")
metric_names_array = np.array(metric_names).reshape(-1, 1)  # Convert to column vector
table_data_with_names = np.hstack((metric_names_array, table_data))
print(table_data_with_names)
