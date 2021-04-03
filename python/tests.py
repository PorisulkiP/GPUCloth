# ТЕСТИРОВАНИЕ

import unittest
import all_file

class PhysicsTestCase(unittest.TestCase):
    """ Тесты для функций velocity и acceleration"""

    def test_velocity(self):
        """ """
        a = [1, 1, 1]
        answer = all_file.Physics.velocity(a)
        self.assertEqual(round(answer, 3), round(2.943, 3))

if __name__ == "__main__":
    unittest.main()

