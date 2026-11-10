import unittest

from src.models import Terrain
from src.pathfinder import astar
from src.utilities import points_in_bounds


class PointsInBoundsTest(unittest.TestCase):
    def test_inclusive_bounds_and_multiple_points(self):
        bounds = ((-2, 3), (4, 7))
        self.assertTrue(points_in_bounds([(-2, 3), (4, 7), (0, 5)], bounds))
        for point in [(-3, 5), (5, 5), (0, 2), (0, 8)]:
            with self.subTest(point=point):
                self.assertFalse(points_in_bounds([(0, 5), point], bounds))

    def test_empty_and_generator_inputs(self):
        self.assertTrue(points_in_bounds([], ((0, 0), (0, 0))))
        self.assertTrue(points_in_bounds(((x, x) for x in range(3)), ((0, 0), (2, 2))))

    def test_pathfinder_preserves_uneven_map_bounds(self):
        terrain = [[Terrain.EMPTY], [Terrain.EMPTY, Terrain.EMPTY]]
        self.assertEqual(astar(terrain, (0, 0), (1, 1)), [(1, 0), (0, 1)])
        for endpoint in [(-1, 0), (2, 0), (0, 1), (1, -1)]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                astar(terrain, (0, 0), endpoint)
        with self.assertRaises(ValueError):
            astar([], (0, 0), (0, 0))

    def test_mask_preserves_order_and_checks_every_point(self):
        points = [(0, 0), (-1, 0), (2, 2), (0, 3), (0, 0)]
        self.assertEqual(
            points_in_bounds(iter(points), ((0, 0), (2, 2)), return_mask=True),
            [True, False, True, False, True],
        )
        self.assertEqual(points_in_bounds([], ((0, 0), (2, 2)), return_mask=True), [])

    def test_boolean_mode_short_circuits(self):
        points = iter([(-1, 0), (0, 0)])
        self.assertFalse(points_in_bounds(points, ((0, 0), (2, 2))))
        self.assertEqual(next(points), (0, 0))
