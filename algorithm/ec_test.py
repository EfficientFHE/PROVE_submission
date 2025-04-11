import unittest
from .elliptic_curve import *
from .config_file import config_BLS12_377

class TestEllipticCurvePointAddition(unittest.TestCase):
    def setUp(self):
        # Set up the configuration for the elliptic curve y^2 = x^3 + 1 over a finite field
        config = {
            'prime': 23,          # A small prime field for simplicity
            'order': 19,          # Arbitrary order for testing
            'a': 0,               # Coefficient a = 0
            'b': 1,               # Coefficient b = 1
            'generator': [1, 1]   # Arbitrary generator point
        }
        self.affine_system = ECCWeierstrassAffine(config)
        self.projective_system = ECCWeierstrassProjective(config)
   
    @unittest.skip
    def test_affine_point_addition(self):
        # Create two points on the curve using generate_point
        pointA = self.affine_system.generate_point([3, 10])
        pointB = self.affine_system.generate_point([9, 7])

        # Perform point addition
        result_point = self.affine_system.point_add(pointA, pointB)

        # Expected result manually calculated
        expected_x = FieldEle(17, 23)
        expected_y = FieldEle(20, 23)

        # Verify the result of point addition
        self.assertEqual(result_point[0], expected_x, f"Expected x-coordinate {expected_x} but got {result_point[0]}")
        self.assertEqual(result_point[1], expected_y, f"Expected y-coordinate {expected_y} but got {result_point[1]}")

    @unittest.skip
    def test_projective_point_addition(self):
        # Create two points on the curve using generate_point
        pointA = self.projective_system.generate_point([3, 10, 1])
        pointB = self.projective_system.generate_point([9, 7, 1])



        # Perform point addition in projective coordinates
        result_point = self.projective_system.point_add(pointA, pointB)

        # Convert the result back to affine coordinates
        result_point_affine = self.projective_system.convert_to_affine(result_point)

        # Expected result manually calculated
        expected_x = FieldEle(17, 23)
        expected_y = FieldEle(20, 23)

        # Verify the result of point addition in affine coordinates
        self.assertEqual(result_point_affine[0], expected_x, f"Expected x-coordinate {expected_x} but got {result_point_affine[0]}")
        self.assertEqual(result_point_affine[1], expected_y, f"Expected y-coordinate {expected_y} but got {result_point_affine[1]}")

    def test_large_projective_point_addition(self):
        self.projective_system = ECCWeierstrassProjective(config_BLS12_377)
        
        pac = copy.copy(config_BLS12_377["generator"])
        pac.append(1)

        pbc = copy.copy(config_BLS12_377["generator"])
        pbc.append(1)

        pointA = self.projective_system.generate_point(pac)
        pointB = self.projective_system.generate_point(pbc)

        result_point = self.projective_system.point_add(pointA, pointB)
        result_point_affine = self.projective_system.convert_to_affine(result_point)
        print(result_point_affine.coordinates)
if __name__ == "__main__":
    unittest.main()
