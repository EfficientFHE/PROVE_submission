"""library of finite field operations.

This library is used to implement the finite field operations for the
high-precision
elliptic curve.

- Data Representation.
The high-precision coordinate is represented as a vector of uint16.
The actual base is increased with the index.
E.g.
    index     [   0,     1,     2, ...]
bit precision [0~15, 16~31, 32~47, ...]

It includes Barret/Lazy/RNS based modular multiplication

# Function Name Terminology
## <func>_<in_datatype> indicates that the function only works for a single
## precision.
## <func> indicates that the function works for general bit precision.

# Terminology
## Chunk Reduction: e.g. u8-chunk -> u16-chunk or u32-chunk
## Chunk Decomposition <-> Chunk Merge:
### Chunk Decomposition: break int into multiple low-precision chunks.
### Chunk Merge: Merge multiple low-precision chunks into an int.
"""

import functools

import jax
import jax.numpy as jnp
import util
import numpy as np


total_modulus = util.total_modulus
to_rns = util.to_rns


jax.config.update("jax_enable_x64", True)


@jax.named_call
@functools.partial(jax.jit, static_argnames="chunk_shift_bits")
def check_any_chunk_with_carry(
    value_c: jax.Array,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
) -> jax.Array:
  """This function check whether any chunk of input vector 'value_c' has carry.

  Args:
    value_c: The value to carry add.
    chunk_shift_bits: ideal bit precision of any given chunk. Note that: actual
      bit precision of any given chunk might be higher than chunk_shift_bits
      because it needs to hold the overflow.

  Returns:
    cond: A boolean value indicating whether any chunk of input vector 'value_c'
    has carry.
  """
  high = jnp.right_shift(value_c, chunk_shift_bits)
  cond = jnp.any(jnp.not_equal(high, 0))
  return cond


@jax.named_call
@functools.partial(jax.jit, static_argnames=("mask", "chunk_shift_bits"))
def carry_propagation(
    value_c: jax.Array,
    mask=util.U32_MASK,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
):
  """The purpose of this API is to enable carry propagation.

  Args:
    value_c: The value to carry propagate.
    mask: 2**chunk_bitwidth - 1,
    chunk_shift_bits: chunk_bitwidth

  This function split each chunk into high and low parts, and high part is left
    roll by 1 to carry the overflowed bits to the next chunk.
  Note that: in a given jax.array, bit range of the chunk within the original
    high precision value is increased from left to the right.

  Returns:
    value_c: The value after carry adding.
  """
  precision_dim = value_c.shape[-1]
  roll_mat = jnp.array(
      [0, 1]
      + ([0] * (precision_dim) + [1]) * (precision_dim - 2)
      + [1]
      + [0] * (precision_dim - 1),
      dtype=jnp.uint16,
  ).reshape(precision_dim, precision_dim)
  low = jnp.bitwise_and(value_c, mask)
  high = jnp.right_shift(value_c, chunk_shift_bits).astype(jnp.uint16)
  high = jnp.matmul(high, roll_mat, preferred_element_type=jnp.uint32).astype(
      jnp.uint16
  )
  value_c = jnp.add(low, high)
  return value_c


def conv_1d(value_a: jax.Array, value_b: jax.Array):
  """This function performs a 1D convolution of two u32 arrays.

  Args:
    value_a: The chunk-decomposition representation of the high-precision int.
    value_b: The chunk-decomposition representation of the high-precision int.

  Returns:
    conv: The convolution results of two input arrays being casted to uint8.
  """
  value_a = jax.lax.bitcast_convert_type(value_a, jnp.uint8).reshape(-1)
  value_b = jax.lax.bitcast_convert_type(value_b, jnp.uint8).reshape(-1)
  conv = jnp.convolve(
      value_a,
      value_b,
      preferred_element_type=jnp.uint32,
  )
  return conv


@jax.named_call
@functools.partial(jax.jit, static_argnames=("chunk_num_u16", "chunk_num_u32"))
def rechunkify(mul_result: jax.Array, chunk_num_u16, chunk_num_u32):
  """Given the carry add takes O(C) algorithm complexity, where C is the number of chunks.

  This function performs chunk reduction for ther results of the convolution,
  i.e. merge two consecutive chunks into one chunk with double precision.
  E.g. u8[0, 8, 8, 0] -> u16[8, 2048] 0-> u32[526336]

  Args:
    mul_result: The chunk-wise multiplication (using convolution) result.
    chunk_num_u16: The number of bits in each chunk.
    chunk_num_u32: The number of bits in the second chunk.

  Returns:
    value_c: The result of the chunk reduction.
  """
  shift_0_8_u16x4 = jnp.array(
      [[0, 8] for _ in range(chunk_num_u16 * 4)], dtype=jnp.uint8
  )
  shift_0_16_u32x4 = jnp.array(
      [[0, 16] for _ in range(chunk_num_u32 * 4)], dtype=jnp.uint8
  )
  new_shape = (
      mul_result.shape[:-1] + (-1, 2) if mul_result.ndim == 2 else (-1, 2)
  )
  value_c = mul_result.reshape(new_shape)
  value_c = jnp.left_shift(value_c, shift_0_8_u16x4[:chunk_num_u16])
  value_c = jnp.sum(value_c, axis=-1)
  value_c = value_c.reshape(new_shape).astype(jnp.uint64)
  value_c = jnp.left_shift(value_c, shift_0_16_u32x4[:chunk_num_u32])
  value_c = jnp.sum(value_c, axis=-1)
  return value_c


@jax.named_call
@functools.partial(jax.jit, static_argnames="chunk_num")
def compare_u32(
    value_a: jax.Array, value_b: jax.Array, chunk_num=util.U32_CHUNK_NUM
):
  """Compare two u32 values.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    chunk_num_u32: The number of chunks in the u32 value.

  Returns:
  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  sign = jnp.sign(
      jnp.subtract(value_a.astype(jnp.int64), value_b.astype(jnp.int64))
  )
  comp_check_vec_weights = jnp.array(
      [2**i for i in range(chunk_num)], dtype=jnp.int32
  )
  weight = jnp.multiply(sign, comp_check_vec_weights)
  cond = weight.sum(axis=-1)
  return cond


def add_2u32(value_a: jax.Array, value_b: jax.Array):
  """Add two u32 values.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.

  Returns:
    value_c: The result of the addition.
  """
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint64), value_b.astype(jnp.uint64)
  )
  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  # value_c = carry_propagation_scan_u32(value_c)
  return value_c.astype(jnp.uint32)


def add_3u32(value_a: jax.Array, value_b: jax.Array, value_d: jax.Array):
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint64), value_b.astype(jnp.uint64)
  )
  value_c = jax.numpy.add(
      value_c.astype(jnp.uint64), value_d.astype(jnp.uint64)
  )
  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  # value_c = carry_propagation_scan_u32(value_c)
  return value_c.astype(jnp.uint32)


@jax.named_call
@functools.partial(jax.jit, static_argnames=("mask", "chunk_num"))
def sub_2u32(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    chunk_num=util.U32_CHUNK_NUM,
):
  """Subtract two u32 values.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    mask: The mask to apply to the value.
    chunk_num_u32: The number of chunks in the u32 value (default: 24).

  Returns:
    value_c: The result of the subtraction.
  """
  borrow_high_pad_zero_array = jnp.array(
      [0] + [1] * (chunk_num - 2) + [0], dtype=jnp.uint64
  )
  borrow_low_array = jnp.array(
      [mask + 1] * (chunk_num - 1) + [0], dtype=jnp.uint64
  )
  value_a = jnp.add(value_a.astype(jnp.uint64), borrow_low_array.astype(jnp.uint64))
  value_c = jnp.subtract(value_a, value_b)
  value_c = jnp.subtract(value_c, borrow_high_pad_zero_array)

  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  # value_c = carry_propagation_scan_u32(value_c)
  if value_c.ndim == 1:
    value_c = value_c.at[chunk_num - 1].set(value_c[chunk_num - 1] - 1)
  else:
    value_c = value_c.at[:, chunk_num - 1].set(
        value_c[:, chunk_num - 1] - 1
    )

  value_c = value_c.astype(jnp.uint32)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int_chunk", "chunk_num")
)
def cond_sub_mod_u32(
    value_a: jax.Array,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    chunk_num=util.U32_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - modulus_377_int.

  Args:
    value_a: The minuend.
    modulus_377_int: The modulus 377.
    chunk_num_u16: The number of chunks in the u32 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  compare_u32_local = functools.partial(
      compare_u32, chunk_num=chunk_num
  )
  sub_2u32_local = functools.partial(sub_2u32, chunk_num=chunk_num)

  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint32)

  cond = compare_u32_local(value_a, modulus_377_int_array)
  value_b = sub_2u32_local(value_a, modulus_377_int_array)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))
  value_c = jnp.where(cond, value_b, value_a)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int_chunk", "chunk_num")
)
def cond_sub_2u32(
    value_a: jax.Array,
    value_b: jax.Array,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    chunk_num=util.U32_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - value_b.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.
    modulus_377_int: The modulus 377.
    chunk_num_u16: The number of chunks in the u32 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint32)
  compare_u32_local = functools.partial(
      compare_u32, chunk_num=chunk_num
  )
  sub_2u32_local = functools.partial(sub_2u32, chunk_num=chunk_num)

  cond = compare_u32_local(value_a, value_b)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))

  value_ap = jnp.add(
      value_a.astype(jnp.uint64), modulus_377_int_array.astype(jnp.uint64)
  )

  value_a = jnp.where(cond, value_a.astype(jnp.uint64), value_ap)
  value_c = sub_2u32_local(value_a, value_b)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "chunk_num",
        "chunk_shift_bits",
        "output_dtype",
        "vmap_axes",
    ),
)
def mul_2u32(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    chunk_num=util.U32_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
    output_dtype=jnp.uint32,
    vmap_axes=(0, 0),
):
  """Multiply two u32 values.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u32 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value.
    output_dtype: The desired output data type.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  batch_dim = value_a.shape[0]
  mul_result = jax.vmap(conv_1d, in_axes=vmap_axes)(value_a, value_b)
  mul_result = jnp.pad(mul_result, ((0, 0), (0, 1)))
  value_c = rechunkify(
      mul_result, 4 * chunk_num, 2 * chunk_num
  )

  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  # value_c = carry_propagation_scan_u32(value_c)
  ratio = 16 if output_dtype == jnp.uint8 else 2
  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), output_dtype
  ).reshape(batch_dim, -1)[:, : ratio * chunk_num]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "barrett_shift_u8",
        "chunk_num_u16",
        "chunk_num_u32",
        "vmap_axes",
    ),
)
def mul_shift_2u32x2x1(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    barrett_shift_u8=util.BARRETT_SHIFT_U8,
    chunk_num_u16=util.U16_CHUNK_NUM,
    chunk_num_u32=util.U32_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
    vmap_axes=(0, None),
):
  """Multiply and shift two u32 values.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    barrett_shift_u8: The number of bits to shift the value.
    chunk_num_u16: The number of chunks in the u16 value.
    chunk_num_u32: The number of chunks in the u32 value.
    vmap_axes: (0, None) means axis 0 is the mapped access,
                and The rest is not.

  Returns:

  """
  batch_dim = value_a.shape[0]
  conv = jax.vmap(conv_1d, in_axes=vmap_axes)(value_a, value_b)
  conv = jnp.pad(conv, ((0, 0), (0, 1)))
  value_c = rechunkify(
      conv, chunk_num_u16 * 3, chunk_num_u32 * 3
  )
  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  # value_c = carry_propagation_scan_u32(value_c)
  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), jnp.uint8
  ).reshape(batch_dim, -1)[:, barrett_shift_u8:]
  value_c = jax.lax.bitcast_convert_type(
      jnp.pad(value_c, ((0, 0), (0, 3))).reshape(batch_dim, -1, 4), jnp.uint32
  )[:, :chunk_num_u32]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_377_int_chunk",
        "mu_377_int_chunk",
        "chunk_num",
        "vmap_axes",
    ),
)
def mod_mul_barrett_2u32(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    mu_377_int_chunk=util.MU_377_INT_CHUNK,
    chunk_num=util.U32_CHUNK_NUM,
    vmap_axes=(0, None),
):
  """Multiply two u32 values with Barrett reduction.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    mask: The mask to apply to the value.
    modulus_377_int: The modulus 377.
    mu_377_int: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u32 value (default: 24).

  Returns:
    value_c: The result of the multiplication.
  """
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint32)
  mu_377_int_array = jnp.asarray(mu_377_int_chunk, jnp.uint32)

  mul_2u32_const = functools.partial(mul_2u32, vmap_axes=vmap_axes)
  sub_2u32_const = functools.partial(
      sub_2u32, mask=mask, chunk_num=chunk_num * 2
  )
  value_x = mul_2u32(value_a, value_b)
  value_d = mul_shift_2u32x2x1(value_x, mu_377_int_array)
  value_e = mul_2u32_const(value_d, modulus_377_int_array)
  value_t = sub_2u32_const(value_x, value_e)
  value_c = cond_sub_mod_u32(value_t[:, :chunk_num])
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "modulus_lazy_mat",
        "mask",
        "chunk_num_u8",
        "chunk_shift_bits",
    ),
)
def mod_mul_lazy_2u32(
    value_a,
    value_b,
    modulus_lazy_mat=util.MODULUS_377_LAZY_MAT,
    mask=util.U32_MASK,
    chunk_num_u8=util.U8_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
):
  """Multiply two u32 values with lazy matrix reduction.

  Args:
    value_a: The first u32 value.
    value_b: The second u32 value.
    modulus_lazy_mat: The lazy matrix.
    mask: The mask to apply to the value.
    chunk_num_u8: The number of chunks in the u8 value.
    chunk_shift_bits: The number of bits to shift the value.

  Returns:
    value_c: The result of the multiplication.
  """
  batch_dim = value_a.shape[0]
  modulus_lazy_mat = jnp.asarray(modulus_lazy_mat, dtype=jnp.uint16)
  mul_2u8 = functools.partial(
      mul_2u32,
      mask=mask,
      chunk_num=util.U32_EXT_CHUNK_NUM,
      chunk_shift_bits=chunk_shift_bits,
      output_dtype=jnp.uint8,
  )
  value_c = mul_2u8(value_a, value_b)
  standard_product_low = value_c[:, :chunk_num_u8]
  standard_product_high = value_c[:, chunk_num_u8:chunk_num_u8*2+4]

  reduced = jnp.matmul(
      standard_product_high.astype(jnp.uint16),
      modulus_lazy_mat.astype(jnp.uint16),
      preferred_element_type=jnp.uint32,
  )
  value_c_reduced = jnp.add(
      standard_product_low.astype(jnp.uint32), reduced.astype(jnp.uint32)
  )
  value_c_reduced_u32 = rechunkify(
      value_c_reduced, chunk_num_u8 // 2, chunk_num_u8 // 4
  )
  value_c_reduced_u32 = jnp.pad(value_c_reduced_u32, ((0, 0), (0, 1)))

  value_c_carried = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c_reduced_u32
  )
  # value_c_carried = carry_propagation_scan_u32(value_c_reduced_u32)

  value_c_u16 = jax.lax.bitcast_convert_type(
      value_c_carried.astype(jnp.uint32), jnp.uint32
  ).reshape(batch_dim, -1)[:, : util.U32_EXT_CHUNK_NUM]
  return value_c_u16
