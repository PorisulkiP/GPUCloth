# ТЕСТИРОВАНИЕ

import unittest

class PhysicsTestCase(unittest.TestCase):
    """ Тесты для функций velocity и acceleration"""

    def test_find_velocity(self):
        """ """
        a = [1, 1, 1]
        g = [0,0,-9.8]
        fps = 24
        V = [a[i]-(a[i] - (g[i]/(60/fps))) for i in range(0, 3)]

        self.assertEqual(round(V[2], 2), round(-3.92, 2))

    def test_find_acceleration(self):
         
        x = [2, 2, 2]
        g = [0, 0, -9.8]
        fps = 24
        Vx = [x[i]-(x[i] - (g[i]/(60/fps))) for i in range(0, 3)]
        V2 = [Vx[i]-(Vx[i] - (g[i]/(60/fps))) for i in range(0, 3)]
        
        a = [(Vx[i] - V2[i])/(60/fps) for i in range(0, 3)]

        self.assertEqual(round(a[2], 8), round(0, 2))

    def test_cloth_deformation(self):

        m = 10
        g = [0, 0, 9.8]
        fps = 24
        a = 2

        p = [((m[i] * (g[i] - a[i])) / fps) for i in range(0, )]

        self.assertEqual(round(p[2], 8), round(0, 8))
    
if __name__ == "__main__":
    unittest.main()