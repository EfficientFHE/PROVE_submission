"""Pippenger algorithm for elliptic curves.

This module implements the Pippenger algorithm for elliptic curves. The
algorithm
is a generalization of the elliptic curve algorithm that can be used to find
elliptic curves of arbitrary order.

The Pippenger algorithm works by first finding a set of "scalars" and "points"
that lie on the elliptic curve. These scalars and points are then used to
construct
a "bucket" for each window of the elliptic curve. The buckets are then reduced
to a single point for each window. Finally, the points from all of the windows
are merged together to form the final elliptic curve.

The Pippenger algorithm is a powerful tool for finding elliptic curves, and it
has been used to find elliptic curves of arbitrary order and dimension.
"""

import copy
from ctypes import util
import math
from typing import List

import jax
import jax.numpy as jnp
import elliptic_curve as jec
import finite_field as ff
import util
import numpy as np

deepcopy = copy.deepcopy

'''
Example Parameters:
SLICE_LENGTH = 4
WINDOW_NUM = int(math.ceil(255 / SLICE_LENGTH))
BUCKET_NUM_PER_WINDOW = 2**SLICE_LENGTH
MSM_LENGTH = 1024
COORDINATE_NUM = 4
'''


def selective_padd_with_zero(
  partial_sum, single_point, select, is_zero
  ):
    """Padd the partial sum with the single point, but only if the selection state is 1.

    Args:
      partial_sum: The partial sum.
      single_point: The single point.
      select: The selection state.
      is_zero: The zero states.

    Returns:
      The new partial sum.
    """
    _, batch_dim, _ = partial_sum.shape
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point
    )

    cond_select = jnp.equal(select, 1).reshape(1, batch_dim, 1)
    sum_result = jnp.where(cond_select, new_partial_sum, partial_sum)

    cond_zero = jnp.equal(is_zero, 1).reshape(1, batch_dim, 1)
    cond_select_and_zero = jnp.logical_and(cond_select, cond_zero)
    result = jnp.where(cond_select_and_zero, single_point, sum_result)
    return result


def padd_with_zero(partial_sum, single_point, ps_is_zero, sp_is_zero):
    """Padd the partial sum with the single point.

    Check if the partial sum is equal to the single point first.

    Args:
      partial_sum: The partial sum.
      single_point: The single point.
      ps_is_zero: The zero states of the partial sum.
      sp_is_zero: The zero states of the single point.

    Returns:
      The new partial sum.
    """
    _, batch_dim, _ = partial_sum.shape
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point
    )
    cond_sp_zero = jnp.equal(sp_is_zero, 1).reshape(1, batch_dim, 1)
    cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
    result_1 = jnp.where(cond_sp_zero, partial_sum, single_point)
    result_2 = jnp.where(
        jnp.logical_or(cond_sp_zero, cond_ps_zero), result_1, new_partial_sum
    )
    return result_2


def padd_with_zero_alter(partial_sum, single_point, ps_is_zero):

    _, batch_dim, _ = partial_sum.shape
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point
    )
    cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
    result_2 = jnp.where(cond_ps_zero, single_point, new_partial_sum)
    return result_2


def padd_with_zero_and_pdul_check(
    partial_sum, single_point, ps_is_zero, sp_is_zero
  ):
    """Padd the partial sum with the single point.

    Check if the partial sum is equal to the single point first. If they are
    equal, then double the partial sum.

    Args:
      partial_sum: The partial sum.
      single_point: The single point.
      ps_is_zero: The zero states of the partial sum.
      sp_is_zero: The zero states of the single point.

    Returns:
      The new partial sum.
    """
    # coordinate_dim, batch_dim, precision_dim = partial_sum.shape
    _, batch_dim, _ = partial_sum.shape
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point
    )
    double_partial_sum = jec.pdul_lazy_xyzz_pack(partial_sum)
    cond_equal = jnp.all(partial_sum == single_point, axis=(0, 2)).reshape(
        1, batch_dim, 1
    )
    cond_sp_zero = jnp.equal(sp_is_zero, 1).reshape(1, batch_dim, 1)
    cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
    result_1 = jnp.where(cond_sp_zero, partial_sum, single_point)
    result_2 = jnp.where(
        jnp.logical_or(cond_sp_zero, cond_ps_zero), result_1, new_partial_sum
    )
    reuslt_3 = jnp.where(cond_equal, double_partial_sum, result_2)
    return reuslt_3


def bucket_accumulation_algorithm(
                        all_buckets: jnp.ndarray,
                        all_points: jnp.ndarray,
                        selection_list: jnp.ndarray,
                        zero_states_list: jnp.ndarray,
                        msm_length: int):
  """Non-scan version BA."""
  coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
  for i in range(msm_length):
    point = jax.lax.broadcast_in_dim(all_points[i], (coordinate_dim, buckets_dim, precision_dim), (0,2))
    all_buckets = selective_padd_with_zero(all_buckets, point, selection_list[i], zero_states_list[i])
  return all_buckets

def bucket_accumulation_scan_algorithm(
                            all_buckets: jnp.ndarray,
                        all_points: jnp.ndarray,
                        selection_list: jnp.ndarray,
                        zero_states_list: jnp.ndarray,
                        msm_length: int):
  """Scan version BA."""
  coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
  def scan_body(buckets, point_with_cond_pack):
    point, selection, zero_states = point_with_cond_pack
    point = jax.lax.broadcast_in_dim(point, (coordinate_dim, buckets_dim, precision_dim), (0,2))
    all_buckets = selective_padd_with_zero(buckets, point, selection, zero_states)
    return all_buckets, None
  
  all_buckets, _ = jax.lax.scan(scan_body, all_buckets, (all_points, selection_list, zero_states_list), length=msm_length)
  return all_buckets

def bucket_accumulation_index_algorithm(all_buckets: jnp.ndarray,
                                  all_points: jnp.ndarray,
                                  selection_index_list: jnp.ndarray,
                                  zero_states_list: jnp.ndarray,
                                  msm_length: int):
  """Non-scan version BA with index selection."""
  coordinate_dim, window_dim, buckets_dim, precision_dim = all_buckets.shape
  for i in range(msm_length):
    point = jax.lax.broadcast_in_dim(all_points[i], (coordinate_dim, window_dim, precision_dim), (0,2))
    selective_buckets = all_buckets[:, jnp.arange(window_dim) , selection_index_list[i], :]
    selective_zero_states = zero_states_list[i, jnp.arange(window_dim), selection_index_list[i]]  
    selective_update = padd_with_zero_alter(selective_buckets, point, selective_zero_states)
    all_buckets = all_buckets.at[:, jnp.arange(window_dim) , selection_index_list[i], :].set(selective_update)
  return all_buckets


def bucket_accumulation_index_scan_algorithm(all_buckets: jnp.ndarray,
                                  all_points: jnp.ndarray,
                                  selection_index_list: jnp.ndarray,
                                  zero_states_list: jnp.ndarray,
                                  msm_length: int):
  """Scan version BA with index selection."""
  coordinate_dim, window_dim, buckets_dim, precision_dim = all_buckets.shape
  def scan_body(buckets, point_with_cond_pack):
    point, selection_index, zero_states = point_with_cond_pack
    point = jax.lax.broadcast_in_dim(point, (coordinate_dim, window_dim, precision_dim), (0,2))
    selective_buckets = buckets[:, jnp.arange(window_dim) , selection_index, :]
    selective_zero_states = zero_states[jnp.arange(window_dim), selection_index]  
    selective_update = padd_with_zero_alter(selective_buckets, point, selective_zero_states)
    return buckets.at[:, jnp.arange(window_dim) , selection_index, :].set(selective_update), None
  all_buckets, _ = jax.lax.scan(scan_body, all_buckets, (all_points, selection_index_list, zero_states_list), length=msm_length)
  return all_buckets


def bucket_reduction_algorithm(all_buckets: jnp.ndarray,
                         temp_sum: jnp.ndarray,
                         window_sum: jnp.ndarray,
                         bucket_zero_states_list: jnp.ndarray,
                         temp_zero_states_list: jnp.ndarray,
                         window_zero_states_list: jnp.ndarray,
                         bucket_num_in_window: int,):
  """Non-scan version BR."""
  coordinate_dim, window_dim ,buckets_dim, precision_dim = all_buckets.shape
  all_buckets = jnp.flip(all_buckets.transpose(2,0,1,3),axis=0)
  for i in range(bucket_num_in_window - 1):
    temp_sum = padd_with_zero( temp_sum, all_buckets[i], temp_zero_states_list[i], bucket_zero_states_list[i])
    window_sum = padd_with_zero_and_pdul_check( window_sum, temp_sum, window_zero_states_list[i], temp_zero_states_list[i+1])
  return window_sum

def bucket_reduction_scan_algorithm(all_buckets: jnp.ndarray,
                         temp_sum: jnp.ndarray,
                         window_sum: jnp.ndarray,
                         bucket_zero_states_list: jnp.ndarray,
                         temp_zero_states_list: jnp.ndarray,
                         window_zero_states_list: jnp.ndarray,
                         bucket_num_in_window: int,):
  """Scan version BR."""
  coordinate_dim, window_dim ,buckets_dim, precision_dim = all_buckets.shape
  all_buckets = jnp.flip(all_buckets.transpose(2,0,1,3),axis=0)
  def scan_body(temp_and_window_sum_pack, bucket_with_cond_pack):
    temp_sum, window_sum = temp_and_window_sum_pack
    bucket, bucket_zero_states, temp_zero_states, temp_zero_states1, window_zero_states = bucket_with_cond_pack
    temp_sum = padd_with_zero( temp_sum, bucket, temp_zero_states, bucket_zero_states)
    window_sum = padd_with_zero_and_pdul_check( window_sum, temp_sum, window_zero_states, temp_zero_states1)
    return (temp_sum, window_sum), None
  
  (temp_sum, window_sum), _ = jax.lax.scan(scan_body,
                                           (temp_sum, window_sum),
                                           (all_buckets[:bucket_num_in_window-1], 
                                            bucket_zero_states_list[:bucket_num_in_window-1], 
                                            temp_zero_states_list[:bucket_num_in_window-1],
                                            temp_zero_states_list[1:],
                                            window_zero_states_list[:bucket_num_in_window-1]),
                                            length=bucket_num_in_window-1)

  return window_sum


def window_merge_algorithm(window_sum: jnp.ndarray,
                     slice_length: int):
  """Non-scan version WM."""
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  result = window_sum[:, window_dim-1, :].reshape((coordinate_dim, 1, precision_dim))
  for w in range(window_dim - 2, -1, -1):
    for _ in range(slice_length):
      result = jec.pdul_lazy_xyzz_pack(result)
    result = jec.padd_lazy_xyzz_pack(result, window_sum[:, w, :].reshape((coordinate_dim, 1, util.U32_EXT_CHUNK_NUM)))

  result = result.reshape(
      (coordinate_dim, precision_dim)
  )
  return result

def window_merge_scan_algorithm(window_sum: jnp.ndarray,
                     slice_length: int):
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  window_sum = window_sum.transpose(1,0,2)
  result = window_sum[window_dim-1, :, :].reshape((coordinate_dim, 1, precision_dim))
  """Scan version WM."""
  def fori_loop_body(i, result):
    result = jec.pdul_lazy_xyzz_pack(result)
    return result
  def scan_body(result, window_sum):
    result = jax.lax.fori_loop(0, slice_length, fori_loop_body, result)
    result = jec.padd_lazy_xyzz_pack(result, window_sum.reshape((coordinate_dim, 1, util.U32_EXT_CHUNK_NUM)))
    return result, None
  
  result, _ = jax.lax.scan(scan_body, result, window_sum[:window_dim-1, :, :], reverse=True ,length=window_dim-1)
  result = result.reshape(
      (coordinate_dim, precision_dim)
  )
  return result

class MSMPippenger:
  """Pippenger algorithm for elliptic curves.

  Attributes:
    coordinate_num: The number of coordinates in the elliptic curve.
    slice_length: The length of each slice in the elliptic curve.
    window_num: The number of windows in the elliptic curve.
    bucket_num_in_window: The number of buckets in each window.
    slice_mask: The mask for the slices in the elliptic curve.
    blank_point: A JAX array of zeros, used to initialize the buckets.
    all_buckets_jax: A list of lists of JAX arrays, where each array represents
      a bucket.
    selection_list_jax: A list of lists of integers, where each integer
      represents the selection state of a bucket.
    zero_states_list_jax: A list of lists of integers, where each integer
      represents the zero state of a bucket.
    bucket_zero_states_jax: A list of lists of integers, where each integer
      represents the zero state of a bucket.
    temp_sum_zero_states_jax: A list of integers, where each integer represents
      the zero state of a bucket.
    window_sum_zero_states_jax: A list of integers, where each integer
      represents the zero state of the window sum.
    all_points_jax: A JAX array of all the points in the elliptic curve.
    scalars: A list of integers, where each integer represents an Orignal scalar
      from the trace.
    points: A list of JAX arrays, where each array represents an Orignal point
      from the trace.
    window_sum: A JAX array of the window sum.
    msm_length: The length of the MSM trace.
    result: The final elliptic curve.
    lazy_mat: The lazy matrix used for padding and doubling.
  """

  def __init__(self, slice_length):
    self.coordinate_num = util.COORDINATE_NUM

    self.slice_length = slice_length
    self.window_num = int(math.ceil(util.SCALAR_BITS / self.slice_length)) #
    self.bucket_num_per_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_per_window - 1
    self.blank_point = util.int_list_to_array(
        [0] * self.coordinate_num, util.BASE, util.U32_EXT_CHUNK_NUM
    )


    """
    For index selection version
    The same functionality as all_buckets_jax.
    all_buckets_jax and all_buckets2_jax can keep one of them when method decided. 
    """
    self.all_buckets = (
        jnp.array([
            [self.blank_point for _ in range(self.bucket_num_per_window)]
            for _ in range(self.window_num)
        ])
        .reshape((
            self.window_num,
            self.bucket_num_per_window,
            self.coordinate_num,
            util.U32_EXT_CHUNK_NUM,
        ))
        .transpose(2, 0, 1, 3)
    )

    self.window_sum: jnp.ndarray

    self.msm_length = 0
    self.zero_states_list: jnp.ndarray
    self.selection_list: jnp.ndarray
    self.all_points: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)

    self.result = None

  def initialize(self, scalars, points):
    """Initialize the Pippenger algorithm.

    Args:
      scalars: A list of integers, where each integer represents an Orignal
        scalar from the trace.
      points: A list of JAX arrays, where each array represents an Orignal point
        from the trace.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.msm_length = len(scalars)

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        util.int_list_to_array(
            coordinates + [1, 1], util.BASE, util.U32_EXT_CHUNK_NUM
        )
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points = jnp.array(self.points)

    # For BA
    zero_states_pylist, selection_pylist, selection_index_pylist = (
        self.construct_ba_zero_states_and_selection()
    )
    self.zero_states_list = jnp.array(zero_states_pylist, dtype=jnp.uint8).reshape(
        (-1, self.window_num * self.bucket_num_per_window)
    )
    self.selection_list = jnp.array(selection_pylist,dtype=jnp.uint8).reshape(
        (-1, self.window_num * self.bucket_num_per_window)
    )
    # For index selection version BA
    self.zero_states_list2 = jnp.array(zero_states_pylist, dtype=jnp.uint8)
    self.selection_index_list = jnp.array(selection_index_pylist, dtype=jnp.uint32)

    # For BR
    bucket_zero_states_py, temp_sum_zero_states_py, window_sum_zero_states_py = (
        self.construct_br_zero_states(
            zero_states_pylist[len(zero_states_pylist) - 1]
        )
    )
    self.bucket_zero_states = jnp.array(bucket_zero_states_py, dtype=jnp.uint8)
    self.temp_sum_zero_states = jnp.array(temp_sum_zero_states_py, dtype=jnp.uint8)
    self.window_sum_zero_states = jnp.array(window_sum_zero_states_py, dtype=jnp.uint8)

  
  def bucket_accumulation(self, bucket_accumulation_index_algorithm):
    """"BA index selection version"""
    self.all_buckets = bucket_accumulation_index_algorithm(self.all_buckets,
                                                              self.all_points,
                                                              self.selection_index_list,
                                                              self.zero_states_list2[:self.msm_length])
    
    return self.all_buckets

  
  def bucket_reduction(self, bucket_reduction_algorithm):
    """Reduce the buckets to a single point for each window.
    """
    temp_sum = jnp.array([self.blank_point for _ in range(self.window_num)]).transpose(1,0,2)
    window_sum = jnp.array([self.blank_point for _ in range(self.window_num)]).transpose(1,0,2)
    self.window_sum = bucket_reduction_algorithm(
                                      self.all_buckets,
                                      temp_sum,
                                      window_sum,
                                      self.bucket_zero_states[:self.bucket_num_per_window],
                                      self.temp_sum_zero_states[:self.bucket_num_per_window],
                                      self.window_sum_zero_states[:self.bucket_num_per_window])
    return self.window_sum

  def window_merge(self, window_merge_algorithm):
    """Merge the windows to form the final elliptic curve."""
    self.result = window_merge_algorithm(self.window_sum)
    return self.result

  def construct_ba_zero_states_and_selection(self):
    """Construct the zero states and selection for the bucket accumulation (BA) step.

    Returns:
      A tuple of two lists: the zero states for the bucket accumulation, and the
      selection for the bucket accumulation.
    """
    zero_states = [
        deepcopy([1] * self.bucket_num_per_window)
        for _ in range(self.window_num)
    ]
    zero_states_list = []
    zero_states_list.append(deepcopy(zero_states))
    selection_list = []
    selection_index_list = [] # Used for index selection
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection = [
          deepcopy(([0] * self.bucket_num_per_window))
          for _ in range(self.window_num)
      ]
      selection_index = []
      for w in range(self.window_num):
        slice_index = (scalar >> (w * self.slice_length)) & self.slice_mask
        zero_states[w][slice_index] = 0
        selection[w][slice_index] = 1
        selection_index.append(slice_index)

      selection_list.append(deepcopy(selection))
      zero_states_list.append(deepcopy(zero_states))
      selection_index_list.append(deepcopy(selection_index))
    return zero_states_list, selection_list, selection_index_list

  def construct_br_zero_states(self, bucket_zero_states):
    """Construct the zero states for the bucket reduction (BR) step.

    Args:
      bucket_zero_states: The zero states of the buckets.

    Returns:
      A tuple of three lists: the zero states for the bucket reduction, the zero
      states for the temporary sum, and the zero states for the window sum.
    """
    temp_sum_zero_states = np.array([1] * self.window_num)
    window_sum_zero_states = np.array([1] * self.window_num)
    temp_sum_zero_states_list = []
    window_sum_zero_states_list = []
    temp_sum_zero_states_list.append(temp_sum_zero_states)
    window_sum_zero_states_list.append(window_sum_zero_states)
    bucket_zero_states_list = np.flip(
        np.array(bucket_zero_states).transpose(1, 0), axis=0
    )
    for b in range(self.bucket_num_per_window):
      next_temp_sum_zero_states = (
          temp_sum_zero_states_list[b] & bucket_zero_states_list[b]
      )
      next_window_sum_zero_states = (
          window_sum_zero_states_list[b] & next_temp_sum_zero_states
      )
      temp_sum_zero_states_list.append(next_temp_sum_zero_states)
      window_sum_zero_states_list.append(next_window_sum_zero_states)
    return (
        bucket_zero_states_list,
        temp_sum_zero_states_list,
        window_sum_zero_states_list,
    )
  

def padd(partial_sum, single_point):
  return jec.padd_lazy_twisted_pack(partial_sum, single_point)

def padd_with_pdul_check(
    partial_sum, single_point
  ):
    # coordinate_dim, batch_dim, precision_dim = partial_sum.shape
    _, batch_dim, _ = partial_sum.shape
    new_partial_sum = jec.padd_lazy_twisted_pack(
        partial_sum, single_point
    )
    double_partial_sum = jec.pdul_lazy_twisted_pack(partial_sum)
    cond_equal = jnp.all(partial_sum == single_point, axis=(0, 2)).reshape(
        1, batch_dim, 1
    )
    return jnp.where(cond_equal, double_partial_sum, new_partial_sum)



def bucket_accumulation_index_scan_parallel_algorithm_twisted(all_buckets: jnp.ndarray,
                                  all_points: jnp.ndarray,
                                  selection_index_list: jnp.ndarray,
                                  msm_length: int):
  """Scan version BA with index selection."""
  coordinate_dim, batch_window_dim, buckets_dim, precision_dim = all_buckets.shape
  _, _, parallel_dim, _ = all_points.shape #(serial_dim, coordinate_dim, parallel_dim, precision_dim)
  single_window_dim = batch_window_dim // parallel_dim
  def scan_body(buckets, point_with_cond_pack):
    point, selection_index= point_with_cond_pack
    point = jax.lax.broadcast_in_dim(point, (coordinate_dim, parallel_dim, single_window_dim, precision_dim), (0, 1, 3))
    point = point.reshape((coordinate_dim, batch_window_dim, precision_dim))
    selective_buckets = buckets[:, jnp.arange(batch_window_dim) , selection_index, :]
    selective_update = padd(selective_buckets, point)
    return buckets.at[:, jnp.arange(batch_window_dim) , selection_index, :].set(selective_update), None
  
  all_buckets, _ = jax.lax.scan(scan_body, all_buckets, (all_points, selection_index_list), length=msm_length)
  return all_buckets



def bucket_reduction_scan_algorithm_twisted(all_buckets: jnp.ndarray,
                         temp_sum: jnp.ndarray,
                         window_sum: jnp.ndarray,
                         bucket_num_in_window: int,):
  """Scan version BR."""
  coordinate_dim, window_dim ,buckets_dim, precision_dim = all_buckets.shape
  # all_buckets = jnp.flip(all_buckets.transpose(2,0,1,3),axis=0)
  all_buckets = all_buckets.transpose(2,0,1,3)
  def scan_body(temp_and_window_sum_pack, buckets):
    temp_sum, window_sum = temp_and_window_sum_pack
    temp_sum = padd( temp_sum, buckets)
    # window_sum = padd_with_pdul_check(window_sum, temp_sum)
    window_sum = padd(window_sum, temp_sum)
    return (temp_sum, window_sum), None
  
  (temp_sum, window_sum), _ = jax.lax.scan(scan_body,
                                           (temp_sum, window_sum),
                                           all_buckets[:bucket_num_in_window], 
                                          length=bucket_num_in_window,
                                          reverse=True)
  return window_sum


def batch_window_summation_algorithm_twisted(batch_window_sum: jnp.ndarray, all_window_sum: jnp.ndarray, point_parallel: int):
  coordinate_dim, batch_window_dim, precision_dim = all_window_sum.shape
  all_window_sum = all_window_sum.reshape((coordinate_dim, point_parallel, -1, precision_dim)).transpose(1,0,2,3)
  def scan_body(batch_window_sum, single_window_sum):
    batch_window_sum = padd(batch_window_sum, single_window_sum)
    return batch_window_sum, None
  batch_window_sum, _ = jax.lax.scan(scan_body, batch_window_sum, all_window_sum, length=point_parallel)
  return batch_window_sum


def window_merge_scan_algorithm_twisted(window_sum: jnp.ndarray,
                     slice_length: int):
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  window_sum = window_sum.transpose(1,0,2).reshape(
    (window_dim, coordinate_dim, 1, precision_dim))
  result = window_sum[window_dim-1]
  """Scan version WM."""
  def fori_loop_body(i, result):
    result = jec.pdul_lazy_twisted_pack(result)
    return result
  def scan_body(result, window_sum):
    result = jax.lax.fori_loop(0, slice_length, fori_loop_body, result)
    result = jec.padd_lazy_twisted_pack(result, window_sum)
    return result, None
  result, _ = jax.lax.scan(scan_body, result, window_sum[:window_dim-1], reverse=True ,length=window_dim-1)
  result = result.reshape(
      (coordinate_dim, precision_dim)
  )
  return result


class MSMPippengerTwisted:

  def __init__(self, slice_length:int , point_parallel: int):
    self.coordinate_num = util.COORDINATE_NUM

    self.slice_length = slice_length
    self.point_parallel = point_parallel
    self.window_num = int(math.ceil(util.SCALAR_BITS / self.slice_length)) #
    self.batch_window_num = self.window_num * self.point_parallel
    self.bucket_num_per_window = 2**self.slice_length - 1 # Note: here remove the bucket_0
    self.slice_mask = 2**self.slice_length - 1
    self.blank_point = util.int_list_to_array(
        [0, 1, 1, 0], util.BASE, util.U32_EXT_CHUNK_NUM
    )


    """
    For index selection version
    The same functionality as all_buckets_jax.
    all_buckets_jax and all_buckets2_jax can keep one of them when method decided. 
    """
    self.all_buckets = (
        jnp.array([
            [self.blank_point for _ in range(self.bucket_num_per_window)]
            for _ in range(self.window_num)
        ])
        .reshape((
            self.window_num,
            self.bucket_num_per_window,
            self.coordinate_num,
            util.U32_EXT_CHUNK_NUM,
        ))
        .transpose(2, 0, 1, 3)
    )
    coordinate_dim, window_dim ,buckets_dim, precision_dim = self.all_buckets.shape
    self.all_buckets = jnp.tile(self.all_buckets, (1, self.point_parallel, 1, 1))

    self.window_sum: jnp.ndarray

    self.msm_length = 0
    
    self.zero_states_list: jnp.ndarray
    self.selection_list: jnp.ndarray
    self.all_points: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)

    self.temp_sum = jnp.array([self.blank_point for _ in range(self.batch_window_num)]).transpose(1,0,2)
    self.window_sum = jnp.array([self.blank_point for _ in range(self.batch_window_num)]).transpose(1,0,2)
    self.batch_window_sum = jnp.array([self.blank_point for _ in range(self.window_num)]).transpose(1,0,2)

    self.result = None

  def initialize(self, scalars, points):
    """Initialize the Pippenger algorithm.

    Args:
      scalars: A list of integers, where each integer represents an Orignal
        scalar from the trace.
      points: A list of JAX arrays, where each array represents an Orignal point
        from the trace.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.msm_length = len(scalars)

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        util.int_list_to_array(
            coordinates, util.BASE, util.U32_EXT_CHUNK_NUM
        )
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points = jnp.array(self.points)
    serial_dim, coordinate_dim, precision_dim = self.all_points.shape

    # For BA
    selection_index_pylist = self.construct_ba_selection()
    # Note: it contains uint(-1) for the bucket_0.
    # In BA, it may cause some undefined behavior when do bucket selection
    # Currecntly it is correct, because when setting buckets after the computation,
    # jax.numpy will ignore the index with uint(-1) out of index.
    self.selection_index_list = jnp.array(selection_index_pylist).astype(jnp.uint32)
    _, window_dim = self.selection_index_list.shape

    # Batch construction
    self.all_points = self.all_points.reshape((-1, self.point_parallel, coordinate_dim, precision_dim)).transpose(0, 2, 1, 3)
    self.selection_index_list = self.selection_index_list.reshape((-1, window_dim * self.point_parallel))


  def bucket_accumulation(self, bucket_accumulation_index_algorithm):
    """"BA index selection version"""
    self.all_buckets = bucket_accumulation_index_algorithm(self.all_buckets,
                                                              self.all_points,
                                                              self.selection_index_list)
    
    return self.all_buckets

  
  def bucket_reduction(self, bucket_reduction_algorithm):
    """Reduce the buckets to a single point for each window.
    """

    self.window_sum = bucket_reduction_algorithm(
                                      self.all_buckets,
                                      self.temp_sum,
                                      self.window_sum)
    return self.window_sum
  
  def batch_window_summation(self, batch_window_summation_algorithm):
    self.batch_window_sum = batch_window_summation_algorithm(
        self.batch_window_sum,
        self.window_sum
    )
    return self.batch_window_sum

  def window_merge(self, window_merge_algorithm):
    """Merge the windows to form the final elliptic curve."""
    self.result = window_merge_algorithm(self.batch_window_sum)
    return self.result

  def construct_ba_selection(self):
    selection_index_list = [] # Used for index selection
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection_index = []
      for w in range(self.window_num):
        slice_index = ((scalar >> (w * self.slice_length)) & self.slice_mask) - 1
        selection_index.append(slice_index)
      selection_index_list.append(deepcopy(selection_index))
    return selection_index_list
  

def padd_with_sign(partial_sum, single_point, sign):
  neg_single_point = jec.pneg_lazy_twisted_pack(single_point)
  _, batch_dim, _ = partial_sum.shape
  cond_neg = jnp.equal(sign, 1).reshape(1, batch_dim, 1)
  signed_point = jnp.where(cond_neg, neg_single_point, single_point)
  result = jec.padd_lazy_twisted_pack(partial_sum, signed_point)
  return result

def bucket_accumulation_signed_index_scan_parallel_algorithm_twisted(all_buckets: jnp.ndarray,
                                  all_points: jnp.ndarray,
                                  selection_index_list: jnp.ndarray,
                                  selection_sign_list: jnp.ndarray,
                                  msm_length: int):
  """Scan version BA with index selection."""
  coordinate_dim, batch_window_dim, buckets_dim, precision_dim = all_buckets.shape
  _, _, parallel_dim, _ = all_points.shape #(serial_dim, coordinate_dim, parallel_dim, precision_dim)
  single_window_dim = batch_window_dim // parallel_dim
  def scan_body(buckets, point_with_cond_pack):
    point, selection_index, selection_sign= point_with_cond_pack
    point = jax.lax.broadcast_in_dim(point, (coordinate_dim, parallel_dim, single_window_dim, precision_dim), (0, 1, 3))
    point = point.reshape((coordinate_dim, batch_window_dim, precision_dim))
    selective_buckets = buckets[:, jnp.arange(batch_window_dim) , selection_index, :]
    selective_update = padd_with_sign(selective_buckets, point, selection_sign)
    return buckets.at[:, jnp.arange(batch_window_dim) , selection_index, :].set(selective_update), None
  
  all_buckets, _ = jax.lax.scan(scan_body, all_buckets, (all_points, selection_index_list, selection_sign_list), length=msm_length)
  return all_buckets


class MSMPippengerTwistedSigned:

  def __init__(self, slice_length:int , point_parallel: int):
    self.coordinate_num = util.COORDINATE_NUM

    self.slice_length = slice_length
    self.point_parallel = point_parallel
    self.window_num = int(math.ceil(util.SCALAR_BITS / self.slice_length)) #
    self.batch_window_num = self.window_num * self.point_parallel
    self.bucket_num_per_window = 2**(self.slice_length - 1)
    self.slice_mask = 2**self.slice_length - 1
    self.blank_point = util.int_list_to_array(
        [0, 1, 1, 0], util.BASE, util.U32_EXT_CHUNK_NUM
    )


    """
    For index selection version
    The same functionality as all_buckets_jax.
    all_buckets_jax and all_buckets2_jax can keep one of them when method decided. 
    """
    self.all_buckets = (
        jnp.array([
            [self.blank_point for _ in range(self.bucket_num_per_window)]
            for _ in range(self.window_num)
        ])
        .reshape((
            self.window_num,
            self.bucket_num_per_window,
            self.coordinate_num,
            util.U32_EXT_CHUNK_NUM,
        ))
        .transpose(2, 0, 1, 3)
    )
    coordinate_dim, window_dim ,buckets_dim, precision_dim = self.all_buckets.shape
    self.all_buckets = jnp.tile(self.all_buckets, (1, self.point_parallel, 1, 1))

    self.window_sum: jnp.ndarray

    self.msm_length = 0
    
    self.zero_states_list: jnp.ndarray
    self.selection_list: jnp.ndarray
    self.all_points: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)

    self.temp_sum = jnp.array([self.blank_point for _ in range(self.batch_window_num)]).transpose(1,0,2)
    self.window_sum = jnp.array([self.blank_point for _ in range(self.batch_window_num)]).transpose(1,0,2)
    self.batch_window_sum = jnp.array([self.blank_point for _ in range(self.window_num)]).transpose(1,0,2)
    
    self.result = None

  def initialize(self, scalars, points):
    """Initialize the Pippenger algorithm.

    Args:
      scalars: A list of integers, where each integer represents an Orignal
        scalar from the trace.
      points: A list of JAX arrays, where each array represents an Orignal point
        from the trace.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.msm_length = len(scalars)

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        util.int_list_to_array(
            coordinates, util.BASE, util.U32_EXT_CHUNK_NUM
        )
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points = jnp.array(self.points)
    serial_dim, coordinate_dim, precision_dim = self.all_points.shape

    # For BA
    selection_index_pylist, selection_sign_pylist = self.construct_ba_selection_with_sign()
    self.selection_index_list = jnp.asarray(selection_index_pylist).astype(jnp.uint32)
    self.selection_sign_list = jnp.array(selection_sign_pylist, dtype=jnp.uint8)
    _, window_dim = self.selection_index_list.shape

    # Batch construction
    self.all_points = self.all_points.reshape((-1, self.point_parallel, coordinate_dim, precision_dim)).transpose(0, 2, 1, 3)
    self.selection_index_list = self.selection_index_list.reshape((-1, window_dim * self.point_parallel))
    self.selection_sign_list = self.selection_sign_list.reshape((-1, window_dim * self.point_parallel))


  def bucket_accumulation(self, bucket_accumulation_index_algorithm):
    """"BA index selection version"""
    self.all_buckets = bucket_accumulation_index_algorithm(self.all_buckets,
                                                              self.all_points,
                                                              self.selection_index_list,
                                                              self.selection_sign_list,)
    
    return self.all_buckets

  
  def bucket_reduction(self, bucket_reduction_algorithm):
    """Reduce the buckets to a single point for each window.
    """

    self.window_sum = bucket_reduction_algorithm(
                                      self.all_buckets,
                                      self.temp_sum,
                                      self.window_sum)
    return self.window_sum
  
  def batch_window_summation(self, batch_window_summation_algorithm):
    
    self.batch_window_sum = batch_window_summation_algorithm(
       self.batch_window_sum,
        self.window_sum
    )
    return self.batch_window_sum

  def window_merge(self, window_merge_algorithm):
    """Merge the windows to form the final elliptic curve."""
    self.result = window_merge_algorithm(self.batch_window_sum)
    return self.result

  def construct_ba_selection_with_sign(self):
    selection_index_list = [] # Used for index selection
    selection_sign_list = []
    slice_max = 2**self.slice_length
    slice_half = 2**(self.slice_length - 1)
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection_index = []
      selection_sign = []
      carry = 0
      for w in range(self.window_num):
        slice_index = ((scalar >> (w * self.slice_length)) & self.slice_mask)
        slice_index = slice_index + carry
        if slice_index >= slice_half:
          slice_index = abs(slice_index - slice_max)
          carry = 1
        else:
          slice_index = slice_index
          carry = 0
        selection_index.append(slice_index - 1)
        selection_sign.append(carry)
      assert(carry == 0)
      selection_index_list.append(deepcopy(selection_index))
      selection_sign_list.append(deepcopy(selection_sign))
    return selection_index_list, selection_sign_list