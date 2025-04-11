import functools

import jax
import jax.numpy as jnp
import util
from algorithm import config_file
import algorithm.elliptic_curve as ec
import elliptic_curve as jec
import finite_field as ff

from absl.testing import absltest
from absl.testing import parameterized


class TestEllipticCurve(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.coordinate_num = 4
    self.batch_size = 1
    self.x1_int_ = 0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76
    self.y1_int_ = 0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283
    self.x2_int_ = 0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C
    self.y2_int_ = 0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4
    self.point_a = [self.x1_int_, self.y1_int_]
    self.point_b = [self.x2_int_, self.y2_int_]
    self.zero_twisted = [0, 1, 1, 0]
    self.ec_sys = ec.ECCSWeierstrassXYZZ(config_file.config_BLS12_377)
    self.point_a_sys = self.ec_sys.generate_point(self.point_a)
    self.point_b_sys = self.ec_sys.generate_point(self.point_b)
    assert int(self.point_a_sys.coordinates[0].value) == self.point_a[0]
    assert int(self.point_a_sys.coordinates[1].value) == self.point_a[1]
    assert int(self.point_b_sys.coordinates[0].value) == self.point_b[0]
    assert int(self.point_b_sys.coordinates[1].value) == self.point_b[1]
    self.true_result_padd = self.point_a_sys + self.point_b_sys
    self.true_result_padd_affine = self.true_result_padd.convert_to_affine()
    self.true_result_pdub_a = self.point_a_sys + self.point_a_sys
    self.true_result_pdub_a_affine = self.true_result_pdub_a.convert_to_affine()
    self.true_result_pdub_b = self.point_b_sys + self.point_b_sys
    self.true_result_pdub_b_affine = self.true_result_pdub_b.convert_to_affine()

  def test_padd_barrett_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    jit_padd_barrett_xyzz_pack = jax.jit(jec.padd_barrett_xyzz_pack)

    result_jax = jit_padd_barrett_xyzz_pack(point_a_jax, point_b_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_jax[0][0], self.true_result_padd[0].get_value())
    self.assertEqual(result_jax[0][1], self.true_result_padd[1].get_value())
    self.assertEqual(result_jax[0][2], self.true_result_padd[2].get_value())

    self.assertEqual(result_jax[0][3], self.true_result_padd[3].get_value())
    # performance measurement
    tasks = [
        (jit_padd_barrett_xyzz_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_barrett_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_jax = jit_pdul_barrett_xyzz_pack(point_a_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_jax[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_jax[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_jax[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_jax[0][3], self.true_result_pdub_a[3].get_value())

    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz_pack_two_no_batch(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_a_jax = jit_pdul_barrett_xyzz_pack(point_a_jax)
    result_a_int = util.jax_point_pack_to_int_point_batch(result_a_jax)

    result_b_jax = jit_pdul_barrett_xyzz_pack(point_b_jax)
    result_b_int = util.jax_point_pack_to_int_point_batch(result_b_jax)

    self.assertEqual(result_a_int[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_a_int[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_a_int[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_a_int[0][3], self.true_result_pdub_a[3].get_value())
    self.assertEqual(result_b_int[0][0], self.true_result_pdub_b[0].get_value())
    self.assertEqual(result_b_int[0][1], self.true_result_pdub_b[1].get_value())
    self.assertEqual(result_b_int[0][2], self.true_result_pdub_b[2].get_value())
    self.assertEqual(result_b_int[0][3], self.true_result_pdub_b[3].get_value())
    
    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
        (jit_pdul_barrett_xyzz_pack, (point_b_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz_pack_two_batch(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    batch_point = jnp.concatenate([point_a_jax, point_b_jax], axis=1)
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_jax = jit_pdul_barrett_xyzz_pack(batch_point)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_int[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_int[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_int[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_int[0][3], self.true_result_pdub_a[3].get_value())
    self.assertEqual(result_int[1][0], self.true_result_pdub_b[0].get_value())
    self.assertEqual(result_int[1][1], self.true_result_pdub_b[1].get_value())
    self.assertEqual(result_int[1][2], self.true_result_pdub_b[2].get_value())
    self.assertEqual(result_int[1][3], self.true_result_pdub_b[3].get_value())

    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (batch_point,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_padd_lazy_xyzz(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))],
        array_size= util.U32_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))],
        array_size= util.U32_EXT_CHUNK_NUM
    )
    # lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)
    jit_padd_lazy_xyzz_pack = jax.jit(jec.padd_lazy_xyzz_pack)
    result_jax = jec.padd_lazy_xyzz_pack(point_a_jax, point_b_jax)
    # result_jax = jit_padd_lazy_xyzz_pack(point_a_jax, point_b_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_padd[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_padd[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_padd[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_padd[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_lazy_xyzz_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_lazy_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pdul_lazy_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))],
        array_size= util.U32_EXT_CHUNK_NUM
    )

    # lazy_mat = jff.construct_lazy_matrix(util.MODULUS_377_INT)
    jit_pdul_lazy_xyzz_pack = jax.jit(jec.pdul_lazy_xyzz_pack)
    result_jax = jit_pdul_lazy_xyzz_pack(point_a_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)
    
    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_pdub_a[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_pdub_a[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_pdub_a[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_pdub_a[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_lazy_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_lazy_xyzz_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_padd_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U32_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U32_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    result_jax = jit_padd_lazy_twisted_pack(point_a_jax, point_b_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()

    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_padd_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_padd_affine[1].get_value(),
    )

  def test_padd_same_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a1 = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_a2 = twisted_ec_sys.twist_int_coordinates(self.point_a)

    point_a1_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a1], array_size=util.U32_EXT_CHUNK_NUM
    )
    point_a2_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a2], array_size=util.U32_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    result_jax = jit_padd_lazy_twisted_pack(point_a1_jax, point_a2_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()

    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_lazy_twisted_pack, (point_a1_jax, point_a2_jax)),
    ]
    profile_name = "jit_padd_lazy_twisted_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pdul_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U32_EXT_CHUNK_NUM
    )

    jit_pdul_lazy_twisted_pack = jax.jit(jec.pdul_lazy_twisted_pack)
    result_jax = jit_pdul_lazy_twisted_pack(point_a_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()
    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_lazy_twisted_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_lazy_twisted_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_pneg_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U32_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U32_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    jit_pneg_lazy_twisted_pack = jax.jit(jec.pneg_lazy_twisted_pack)
    a_plus_b = jit_padd_lazy_twisted_pack(point_a_jax, point_b_jax)
    neg_b = jit_pneg_lazy_twisted_pack(point_b_jax)
    result_jax = jit_padd_lazy_twisted_pack(a_plus_b, neg_b)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()
    self.assertEqual(
        result_affine_point[0].get_value(), self.point_a_sys[0].get_value()
    )
    self.assertEqual(
        result_affine_point[1].get_value(), self.point_a_sys[1].get_value()
    )

    # performance measurement
    tasks = [
        (jit_pneg_lazy_twisted_pack, (point_b_jax,)),
    ]
    profile_name = "jit_pneg_lazy_twisted_pack"
    # util.profile_jax_functions(tasks, profile_name)

  def test_padd_zero_twisted_pack_new_twisted(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U32_EXT_CHUNK_NUM
    )
    point_zero_jax = util.int_point_batch_to_jax_point_pack(
        [self.zero_twisted], array_size=util.U32_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_lazy_twisted_pack),
            name="jit_padd_lazy_twisted_pack",
        ),
    )
    point_c_jax = jit_padd_lazy_twisted_pack(point_a_jax, point_zero_jax)
    project_twist_sum = util.jax_point_pack_to_int_point_batch(point_c_jax)[0]
    project_twist_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    self.assertEqual(
        project_twist_sum_point[0].get_value() % util.MODULUS_377_INT,
        self.point_a[0],
    )
    self.assertEqual(
        project_twist_sum_point[1].get_value() % util.MODULUS_377_INT,
        self.point_a[1],
    )

if __name__ == "__main__":
  absltest.main()
