"""The jaxite_ec implementation of the Elliptic curve operations on TPU.

Detailed algorithms come from the following papers:
xyzz: https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
affine: https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
projective:
https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo

A non-TPU version of the same functions can be found in
jaxite_ec/algorithm/elliptic_curve.py

To test the functionalities of this library, please refer to
jaxite_ec/elliptic_curve_test.py
"""

import functools

import jax
import jax.numpy as jnp
import finite_field
import util


add_3 = finite_field.add_3u32
add_2 = finite_field.add_2u32
sub = finite_field.sub_2u32
cond_sub = finite_field.cond_sub_2u32
cond_sub_mod = finite_field.cond_sub_mod_u32
mod_mul_barrett = finite_field.mod_mul_barrett_2u32
mod_mul_lazy = finite_field.mod_mul_lazy_2u32


# Barrett Reduction Based Functions
@jax.named_call
def padd_barret_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
):
  """PADD-BARRETT elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-BARRETT elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    zz2: The third generator element.
    zzz2: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u1 = mod_mul_barrett(x1, zz2)
  u2 = mod_mul_barrett(x2, zz1)
  s1 = mod_mul_barrett(y1, zzz2)
  s2 = mod_mul_barrett(y2, zzz1)
  zz1_zz2 = mod_mul_barrett(zz1, zz2)
  zzz1_zzz2 = mod_mul_barrett(zzz1, zzz2)

  p = cond_sub(u2, u1)
  r = cond_sub(s2, s1)

  pp = mod_mul_barrett(p, p)
  rr = mod_mul_barrett(r, r)

  ppp = mod_mul_barrett(pp, p)
  q = mod_mul_barrett(u1, pp)
  zz3 = mod_mul_barrett(zz1_zz2, pp)

  ppp_q_2 = add_3(ppp, q, q)
  ppp_q_2 = cond_sub_mod(ppp_q_2)
  ppp_q_2 = cond_sub_mod(ppp_q_2)

  x3 = cond_sub(rr, ppp_q_2)

  q_x3 = cond_sub(q, x3)
  s1_ppp = mod_mul_barrett(s1, ppp)
  zzz3 = mod_mul_barrett(zzz1_zzz2, ppp)

  y3 = mod_mul_barrett(r, q_x3)
  y3 = cond_sub(y3, s1_ppp)

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def pdul_barret_xyzz(
    x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array
):
  """PDUL-BARRET elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::double_general

  This function implements the PDUL-BARRET elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u = add_2(y1, y1)
  u = cond_sub_mod(u)

  x1x1 = mod_mul_barrett(x1, x1)
  v = mod_mul_barrett(u, u)

  w = mod_mul_barrett(u, v)
  s = mod_mul_barrett(x1, v)

  s_2 = add_2(s, s)
  s_2 = cond_sub_mod(s_2)

  m = add_3(x1x1, x1x1, x1x1)
  m = cond_sub_mod(m)
  m = cond_sub_mod(m)

  mm = mod_mul_barrett(m, m)
  w_y1 = mod_mul_barrett(w, y1)
  zz3 = mod_mul_barrett(v, zz1)
  zzz3 = mod_mul_barrett(w, zzz1)

  x3 = cond_sub(mm, s_2)

  s_x3 = cond_sub(s, x3)

  y3 = mod_mul_barrett(m, s_x3)
  y3 = cond_sub(y3, w_y1)

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def pdul_barrett_xyzz_pack(x1_y1_zz1_zzz1: jax.Array):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[0], x1_y1_zz1_zzz1[1], x1_y1_zz1_zzz1[2], x1_y1_zz1_zzz1[3]
  )


@jax.named_call
def padd_barrett_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array
):
  return padd_barret_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
  )


@jax.named_call
def pdul_barrett_xyzz_pack_batch_first(
    x1_y1_zz1_zzz1: jax.Array, transpose=(0, 1, 2)
):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[:, 0],
      x1_y1_zz1_zzz1[:, 1],
      x1_y1_zz1_zzz1[:, 2],
      x1_y1_zz1_zzz1[:, 3],
  ).transpose(transpose[0], transpose[1], transpose[2])


@jax.named_call
def padd_barrett_xyzz_pack_batch_first(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, transpose=(0, 1, 2)
):
  return padd_barret_xyzz(
      x1_y1_zz1_zzz1[:, 0],
      x1_y1_zz1_zzz1[:, 1],
      x1_y1_zz1_zzz1[:, 2],
      x1_y1_zz1_zzz1[:, 3],
      x2_y2_zz2_zzz2[:, 0],
      x2_y2_zz2_zzz2[:, 1],
      x2_y2_zz2_zzz2[:, 2],
      x2_y2_zz2_zzz2[:, 3],
  ).transpose(transpose[0], transpose[1], transpose[2])


# Lazy Reduction Based Functions
@jax.named_call
def padd_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
):
  """PADD-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    zz2: The third generator element.
    zzz2: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )

  u1 = mod_mul_lazy(x1, zz2)
  u2 = mod_mul_lazy(x2, zz1)
  s1 = mod_mul_lazy(y1, zzz2)
  s2 = mod_mul_lazy(y2, zzz1)
  zz1_zz2 = mod_mul_lazy(zz1, zz2)
  zzz1_zzz2 = mod_mul_lazy(zzz1, zzz2)
  
  p = cond_sub_ext(u2, u1)
  r = cond_sub_ext(s2, s1)
  
  pp = mod_mul_lazy(p, p)
  rr = mod_mul_lazy(r, r)
  
  ppp = mod_mul_lazy(pp, p)
  q = mod_mul_lazy(u1, pp)
  zz3 = mod_mul_lazy(zz1_zz2, pp)
  
  # Can be replaced by mod_add_lazy.
  ppp_q_2 = add_3(ppp, q, q)
  ppp_q_2 = cond_sub_mod_ext(ppp_q_2)
  ppp_q_2 = cond_sub_mod_ext(ppp_q_2)
  
  x3 = cond_sub_ext(rr, ppp_q_2)

  q_x3 = cond_sub_ext(q, x3)
  s1_ppp = mod_mul_lazy(s1, ppp)
  zzz3 = mod_mul_lazy(zzz1_zzz2, ppp)
  
  y3 = mod_mul_lazy(r, q_x3)
  y3 = cond_sub_ext(y3, s1_ppp)
  
  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def pdul_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
):
  """PDUL-BARRET elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::double_general

  This function implements the PDUL-BARRET elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  u = add_2(y1, y1)
  u = cond_sub_mod_ext(u)

  x1x1 = mod_mul_lazy(x1, x1)
  v = mod_mul_lazy(u, u)
  
  w = mod_mul_lazy(u, v)
  s = mod_mul_lazy(x1, v)

  s_2 = add_2(s, s)
  s_2 = cond_sub_mod_ext(s_2)

  m = add_3(x1x1, x1x1, x1x1)
  m = cond_sub_mod_ext(m)
  m = cond_sub_mod_ext(m)

  mm = mod_mul_lazy(m, m)
  w_y1 = mod_mul_lazy(w, y1)
  zz3 = mod_mul_lazy(v, zz1)
  zzz3 = mod_mul_lazy(w, zzz1)

  x3 = cond_sub_ext(mm, s_2)

  s_x3 = cond_sub_ext(s, x3)

  y3 = mod_mul_lazy(m, s_x3)
  y3 = cond_sub_ext(y3, w_y1)

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def padd_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array):
  return padd_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
  )


@jax.named_call
def pdul_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array):
  return pdul_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
  )


# Lazy Reduction Based Function
@jax.named_call
@functools.partial(jax.jit, static_argnames="twisted_d_chunk")
def padd_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    z2: jax.Array,
    t2: jax.Array,
    twisted_d_chunk=util.TWIST_D_INT_CHUNK,
):
  """PADD-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::add_general

  This function implements the PADD-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.
    x2: The first generator element.
    y2: The second generator element.
    z2: The third generator element.
    t2: The fourth generator element.
    twisted_d_chunk: The twisted d parameter.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )

  twisted_d = jnp.asarray(twisted_d_chunk, dtype=jnp.uint32)
  twisted_d = jax.lax.broadcast(twisted_d, [x1.shape[0]])

  # a = mod_mul_lazy(x1, x2)
  # b = mod_mul_lazy(y1, y2)
  # d = mod_mul_lazy(z1, z2)
  # c = mod_mul_lazy(t1, t2)
  stack_x1_y1_z1_t1 = jnp.vstack((x1, y1, z1, t1))
  stack_x2_y2_z2_t2 = jnp.vstack((x2, y2, z2, t2))
  stack_a_b_d_c = mod_mul_lazy(stack_x1_y1_z1_t1, stack_x2_y2_z2_t2)
  a, b, d, c = jnp.vsplit(stack_a_b_d_c, 4)

  c = mod_mul_lazy(c, twisted_d)

  # h = add_2(a, b)
  # h = cond_sub_mod_ext(h)
  # g = add_2(d, c)
  # g = cond_sub_mod_ext(g)
  # e1 = add_2(x1, y1)
  # e1 = cond_sub_mod_ext(e1)
  # e2 = add_2(x2, y2)
  # e2 = cond_sub_mod_ext(e2)
  stack_a_c_x1_x2 = jnp.vstack((a, c, x1, x2))
  stack_b_d_y1_y2 = jnp.vstack((b, d, y1, y2))
  stack_h_g_e1_e2 = add_2(stack_a_c_x1_x2, stack_b_d_y1_y2)
  stack_h_g_e1_e2 = cond_sub_mod_ext(stack_h_g_e1_e2)
  h, g, e1, e2 = jnp.vsplit(stack_h_g_e1_e2, 4)

  e = mod_mul_lazy(e1, e2)

  # f = cond_sub_ext(d, c)
  # e = cond_sub_ext(e, h)
  stack_d_e= jnp.vstack((d, e))
  stack_c_h = jnp.vstack((c, h))
  stack_f_e = cond_sub_ext(stack_d_e, stack_c_h)
  f, e = jnp.vsplit(stack_f_e, 2)

  
  # x3 = mod_mul_lazy(e, f)
  # y3 = mod_mul_lazy(g, h)
  # z3 = mod_mul_lazy(f, g)
  # t3 = mod_mul_lazy(h, e)
  stack_e_g_f_h = jnp.vstack((e, g, f, h))
  stack_f_h_g_e = jnp.vstack((f, h, g, e))
  stack_x3_y3_z3_t3 = mod_mul_lazy(stack_e_g_f_h, stack_f_h_g_e)
  x3, y3, z3, t3 = jnp.vsplit(stack_x3_y3_z3_t3, 4)

  return jnp.array([x3, y3, z3, t3])

@jax.named_call
@functools.partial(jax.jit, static_argnames="twisted_d_chunk")
def padd_lazy_twisted__(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    z2: jax.Array,
    t2: jax.Array,
    twisted_d_chunk=util.TWIST_D_INT_CHUNK,
):
  """PADD-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::add_general

  This function implements the PADD-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.
    x2: The first generator element.
    y2: The second generator element.
    z2: The third generator element.
    t2: The fourth generator element.
    twisted_d_chunk: The twisted d parameter.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )

  twisted_d = jnp.asarray(twisted_d_chunk, dtype=jnp.uint32)
  twisted_d = jax.lax.broadcast(twisted_d, [x1.shape[0]])

  a = mod_mul_lazy(x1, x2)
  b = mod_mul_lazy(y1, y2)
  d = mod_mul_lazy(z1, z2)
  c = mod_mul_lazy(t1, t2)
  c = mod_mul_lazy(c, twisted_d)

  h = add_2(a, b)
  h = cond_sub_mod_ext(h)
  e1 = add_2(x1, y1)
  e1 = cond_sub_mod_ext(e1)
  e2 = add_2(x2, y2)
  e2 = cond_sub_mod_ext(e2)
  e = mod_mul_lazy(e1, e2)

  e = cond_sub_ext(e, h)

  f = cond_sub_ext(d, c)
  g = add_2(d, c)
  g = cond_sub_mod_ext(g)

  x3 = mod_mul_lazy(e, f)
  y3 = mod_mul_lazy(g, h)
  z3 = mod_mul_lazy(f, g)
  t3 = mod_mul_lazy(e, h)

  return jnp.array([x3, y3, z3, t3])


def pdul_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
):
  """PDUL-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::double_general

  This function implements the PDUL-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  modulus_377_int_array = jnp.asarray(
      util.MODULUS_377_S16_INT_CHUNK, jnp.uint32
  )

  # a = mod_mul_lazy(x1, x1)
  # b = mod_mul_lazy(y1, y1)
  # ct = mod_mul_lazy(z1, z1)
  stack_x1_y1_z1 = jnp.vstack((x1, y1, z1))
  stack_a_b_ct = mod_mul_lazy(stack_x1_y1_z1, stack_x1_y1_z1)
  a, b, ct = jnp.vsplit(stack_a_b_ct, 3)


  # ct2 = add_2(ct, ct)  #
  # ct2 = cond_sub_mod_ext(ct2)  #
  # et = add_2(x1, y1)  #
  # et = cond_sub_mod_ext(et)  #
  # h = add_2(a, b)
  stack_ct_x1_a = jnp.vstack((ct, x1, a))
  stack_ct_y1_b = jnp.vstack((ct, y1, b))
  stack_ct2_et_h = add_2(stack_ct_x1_a, stack_ct_y1_b)
  stack_ct2_et_h = cond_sub_mod_ext(stack_ct2_et_h)
  ct2, et, h = jnp.vsplit(stack_ct2_et_h, 3)

  h = cond_sub_ext(modulus_377_int_array, h)
  e = mod_mul_lazy(et, et)  #
  e = add_2(e, h)  #
  e = cond_sub_mod_ext(e)  #

  g = cond_sub_ext(b, a)  #
  f = cond_sub_ext(g, ct2)  #

  # x3 = mod_mul_lazy(e, f)  #
  # y3 = mod_mul_lazy(g, h)
  # z3 = mod_mul_lazy(f, g)
  # t3 = mod_mul_lazy(h, e)
  stack_e_g_f_h = jnp.vstack((e, g, f, h))
  stack_f_h_g_e = jnp.vstack((f, h, g, e))
  stack_x3_y3_z3_t3 = mod_mul_lazy(stack_e_g_f_h, stack_f_h_g_e)
  x3, y3, z3, t3 = jnp.vsplit(stack_x3_y3_z3_t3, 4)

  return jnp.array([x3, y3, z3, t3])

def pdul_lazy_twisted__(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
):
  """PDUL-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::double_general

  This function implements the PDUL-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_ext = functools.partial(
      cond_sub,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  cond_sub_mod_ext = functools.partial(
      cond_sub_mod,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )
  modulus_377_int_array = jnp.asarray(
      util.MODULUS_377_S16_INT_CHUNK, jnp.uint32
  )

  a = mod_mul_lazy(x1, x1)
  b = mod_mul_lazy(y1, y1)

  ct = mod_mul_lazy(z1, z1)  #
  ct2 = add_2(ct, ct)  #
  ct2 = cond_sub_mod_ext(ct2)  #

  h = add_2(a, b)
  h = cond_sub_ext(modulus_377_int_array, h)  #

  et = add_2(x1, y1)  #
  et = cond_sub_mod_ext(et)  #
  e = mod_mul_lazy(et, et)  #
  e = add_2(e, h)  #
  e = cond_sub_mod_ext(e)  #

  g = cond_sub_ext(b, a)  #
  f = cond_sub_ext(g, ct2)  #
  x3 = mod_mul_lazy(e, f)  #
  y3 = mod_mul_lazy(g, h)
  z3 = mod_mul_lazy(f, g)
  t3 = mod_mul_lazy(e, h)
  return jnp.array([x3, y3, z3, t3])


def pneg_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
):
  """PDUL-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::double_general

  This function implements the PDUL-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """

  modulus_377_int_array = jnp.asarray(
      util.MODULUS_377_S16_INT_CHUNK, jnp.uint32
  )
  sub_ext = functools.partial(
      sub,
      chunk_num=util.U32_EXT_CHUNK_NUM,
  )

  x2 = sub_ext(modulus_377_int_array, x1)
  y2 = y1
  z2 = z1
  t2 = sub_ext(modulus_377_int_array, t1)

  return jnp.array([x2, y2, z2, t2])


def padd_lazy_twisted_pack(x1_y1_z1_t1: jax.Array, x2_y2_z2_t2: jax.Array):
  return padd_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
      x2_y2_z2_t2[0],
      x2_y2_z2_t2[1],
      x2_y2_z2_t2[2],
      x2_y2_z2_t2[3],
  )


def pdul_lazy_twisted_pack(x1_y1_z1_t1: jax.Array):
  return pdul_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
  )


def pneg_lazy_twisted_pack(x1_y1_z1_t1: jax.Array):
  return pneg_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
  )
