import unittest

from coin_detection import actual_radius_mm, classify_by_diameter, estimate_camera_distance_mm


class CoinDetectionMathTests(unittest.TestCase):
    def test_actual_radius_formula(self):
        radius_mm = actual_radius_mm(radius_pixels=100, camera_distance_mm=500, focal_length_px=1000)
        self.assertAlmostEqual(radius_mm, 50.0)

    def test_distance_estimation_formula(self):
        distance_mm = estimate_camera_distance_mm(
            measured_radius_pixels=80,
            known_radius_mm=11.0,
            focal_length_px=1200,
        )
        self.assertAlmostEqual(distance_mm, 165.0)

    def test_coin_classification_for_known_diameter(self):
        spec = classify_by_diameter(26.5)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.denomination, 10)

    def test_coin_classification_rejects_far_value(self):
        self.assertIsNone(classify_by_diameter(40.0))


if __name__ == "__main__":
    unittest.main()
