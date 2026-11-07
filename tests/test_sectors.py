import unittest

from src.map import extract_from_raw
from src.mission import RAW_MAP, create_sectors, divide_search
from src.models import Sector
from src.models import Objective, ObjectiveType
from src.drone import DroneController


class SectorTest(unittest.TestCase):
    def test_sectors_partition_the_entire_map(self):
        _, terrain, _, _ = extract_from_raw(RAW_MAP)
        for count in (1, 2, 5, 54):
            with self.subTest(count=count):
                sectors = create_sectors(terrain, count)
                self.assertEqual(len(sectors), count)
                self.assertEqual(set(sectors), {sector.sector_id for sector in sectors.values()})
                cells = []
                for sector in sectors.values():
                    (top, left), (bottom, right) = sector.bounds
                    cells.extend(
                        (row, col)
                        for row in range(top, bottom + 1)
                        for col in range(left, right + 1)
                    )
                self.assertEqual(len(cells), 54)
                self.assertEqual(set(cells), {
                    (row, col) for row in range(6) for col in range(9)
                })

    def test_assigns_prepared_sectors_to_nearest_available_drones(self):
        sectors = {
            "west": Sector(sector_id="west", bounds=((0, 0), (5, 4))),
            "east": Sector(sector_id="east", bounds=((0, 5), (5, 8))),
        }
        original = {key: sector.model_copy(deep=True) for key, sector in sectors.items()}
        plans = divide_search(sectors, [(5, 8), (0, 0)])
        self.assertEqual(plans["1"].sector_id, "west")
        self.assertEqual(plans["0"].sector_id, "east")
        self.assertEqual(sectors, original)

    def test_rejects_mismatched_sector_and_drone_counts(self):
        with self.assertRaises(ValueError):
            divide_search({}, [(0, 0)])

    def test_controller_routes_with_sector_dictionary(self):
        _, terrain, _, _ = extract_from_raw(RAW_MAP)
        sectors = {"west": Sector(sector_id="west", bounds=((0, 0), (5, 3)))}
        controller = DroneController("0", terrain, sectors, {})
        controller.update_position((0, 0))
        controller.update_explored_cells({(0, 0)})
        controller.update_objective(Objective(
            drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR, sector_id="west"
        ))
        self.assertTrue(controller.route)
        self.assertEqual(controller.next_action(), (1, 0))


if __name__ == "__main__":
    unittest.main()
