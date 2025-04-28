from typing import Tuple
import numpy as np
import scipy.stats
from rliable import metrics as rli_metrics
from rliable import library as rli_library


def get_aggregate_stat_fn(aggregate_stat: str):
    if aggregate_stat == "iqm":
        return rli_metrics.aggregate_iqm
    elif aggregate_stat == "mean":
        return rli_metrics.aggregate_mean
    else:
        raise ValueError(f"Invalid aggregate stat: {aggregate_stat}")


def compute_aggregate_stat_and_ci_per_task(data: np.ndarray, agg_stat_name: str, return_interval_est: bool):
    '''Computes the aggregate statistic and the bootstrapped CI for each task separately.
    Args:
        data: The input NumPy ndarray of shape (num_runs, num_tasks).
        agg_stat_name: The name of the aggregate statistic to compute ('iqm' or 'mean').
        return_interval_est: Whether to return the bootstrapped CI.
    '''
    assert data.ndim == 2, "Data must be 2D."
    num_runs, num_tasks = data.shape
    aggregate_stat_fn = get_aggregate_stat_fn(agg_stat_name)
    
    if return_interval_est:
        point_ests = []
        interval_ests = []
        for task_idx in range(num_tasks):
            data_dict = {"data": data[:, [task_idx]]}
            point_est, interval_est = rli_library.get_interval_estimates(
                data_dict,
                func=lambda x: np.array([aggregate_stat_fn(x)]),
                reps=25000,
                confidence_interval_size=0.95
            )
            point_ests.append(point_est["data"].squeeze())
            interval_ests.append(interval_est["data"].squeeze())
        point_ests = np.array(point_ests) # shape (num_tasks,)
        interval_ests = np.array(interval_ests) # shape (num_tasks, 2)
        
        return point_ests, interval_ests
    else: # return the aggregate statistic for each task
        point_ests = []
        for task_idx in range(num_tasks):
            point_ests.append(aggregate_stat_fn(data[:, [task_idx]]))
        point_ests = np.array(point_ests) # shape (num_tasks,)
        return point_ests

def compute_aggregate_stat(data: np.ndarray, stat_name: str, agg_dims: Tuple[int, ...]):
    """
    Warning - this code has not been thorougly tested. TODO: test!
    Computes an aggregate statistic over specified dimensions of a NumPy array.
    Handles negative dimension indices (e.g., -1 for the last dimension).

    Args:
        data: The input NumPy ndarray.
        stat_name: The name of the statistic to compute ('mean' or 'iqm').
        agg_dims: A tuple of dimension indices (positive or negative) to aggregate over.

    Returns:
        A NumPy array with the aggregated statistic, where the dimensions
        specified in agg_dims have been removed. The remaining dimensions
        maintain their original relative order.

    Raises:
        ValueError: If input data is empty, agg_dims is empty, contains non-integers,
                    invalid indices (out of bounds after normalization), or duplicate
                    indices (after normalization).
        ValueError: If stat_name is not 'mean' or 'iqm'.
    """
    ### input validation (initial checks)
    if not data.size > 0:
        raise ValueError("Input data array cannot be empty.")

    if not agg_dims: # Check for empty agg_dims
         raise ValueError("agg_dims cannot be empty. Specify dimensions to aggregate.")

    if not all(isinstance(d, int) for d in agg_dims):
        raise ValueError(f"All elements in agg_dims must be integers. Got: {agg_dims}")

    ### Normalize negative dimensions ###
    ndim = data.ndim
    try:
        # Convert negative indices to their positive equivalents
        # Example: if ndim=5, -1 becomes 4, -2 becomes 3, 0 stays 0
        normalized_agg_dims = tuple(d + ndim if d < 0 else d for d in agg_dims)
    except TypeError: # Should be caught by isinstance check, but good practice
        raise ValueError(f"Error normalizing agg_dims. Ensure all elements are integers: {agg_dims}")

    ### input validation (post-normalization) ###
    # Check for duplicates *after* normalization
    # Example: if ndim=5, agg_dims=(0, -5) becomes (0, 0) which is invalid
    if len(set(normalized_agg_dims)) != len(normalized_agg_dims):
        raise ValueError(f"Duplicate dimensions found in agg_dims (after normalization): {agg_dims} -> {normalized_agg_dims}")

    # Check bounds *after* normalization
    # This catches indices that are still out of range (e.g., original index >= ndim or < -ndim)
    if not all(0 <= d < ndim for d in normalized_agg_dims):
        # Identify which original and normalized dimensions are problematic
        invalid_original = [d for i, d in enumerate(agg_dims) if not (0 <= normalized_agg_dims[i] < ndim)]
        invalid_normalized = [d for d in normalized_agg_dims if not (0 <= d < ndim)]
        raise ValueError(
            f"Invalid dimension index found in agg_dims (out of bounds after normalization). "
            f"Problematic original dims: {invalid_original}, "
            f"corresponding normalized dims: {invalid_normalized}. "
            f"Valid range is [0, {ndim-1}] or [{-ndim}, -1]."
        )

    ### compute the aggregate statistic ###
    # Use the 'normalized_agg_dims' tuple from here onwards
    if stat_name == "mean":
        return np.mean(data, axis=normalized_agg_dims)

    elif stat_name == "iqm":
        # compute dimensions to keep and their original order
        all_dims = tuple(range(ndim))
        # Use normalized_agg_dims to determine keep_dims
        keep_dims = tuple(d for d in all_dims if d not in normalized_agg_dims)

        # define the permutation for transpose: keep_dims first, then agg_dims
        # Use normalized_agg_dims for the permutation
        permutation = keep_dims + normalized_agg_dims
        transposed_data = np.transpose(data, axes=permutation)

        # reshape to combine all aggregated dimensions into one (the last one)
        keep_shape = tuple(data.shape[d] for d in keep_dims)
        reshaped_shape = keep_shape + (-1,)
        reshaped_data = transposed_data.reshape(reshaped_shape)

        # compute IQM along the last dimension (the combined aggregate dimension)
        result = scipy.stats.trim_mean(reshaped_data, proportiontocut=0.25, axis=-1)
        return result
    else:
        raise ValueError(f"Statistic '{stat_name}' not implemented.")
