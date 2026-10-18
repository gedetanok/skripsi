import unittest
import cv2
import numpy as np
from src.data.preprocessing import TECHNIQUES, crop_retina, retina_mask, preprocess

class EmptyMaskTests(unittest.TestCase):
    def test_empty_mask_all_techniques(self):
        for technique in TECHNIQUES:
            with self.subTest(technique=technique):
                result = preprocess(np.zeros((32, 32, 3), np.uint8), technique)
                self.assertEqual(result.shape, (224, 224, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertFalse(result.any())

    def test_nonempty_mask_preserves_previous_formula(self):
        img = np.full((80, 80, 3), 100, np.uint8)
        cropped = crop_retina(img)
        mask = retina_mask(cropped)
        self.assertTrue(mask.any())
        for technique, enhance in TECHNIQUES.items():
            with self.subTest(technique=technique):
                expected = cv2.resize(enhance(cropped, mask) * mask[:, :, None], (224, 224), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(preprocess(img, technique), expected)
