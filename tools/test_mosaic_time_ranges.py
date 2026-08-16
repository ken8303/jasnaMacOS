import unittest

from mosaic_time_ranges import (
    active_segment_indices,
    decode_internal,
    selected_ranges,
    segment_ranges,
)


class MosaicTimeRangeTests(unittest.TestCase):
    def test_maps_source_timeline_onto_test_clip(self):
        self.assertEqual(
            selected_ranges("00:14:45-00:15:10", 14 * 60 + 40, 30),
            [(5, 30)],
        )

    def test_intersects_ranges_with_resumable_segment(self):
        ranges = decode_internal("100.000000/140.000000,250.000000/280.000000")
        self.assertEqual(segment_ranges(ranges, 120, 120), [(0, 20)])

    def test_clean_segment_has_no_ranges(self):
        self.assertEqual(segment_ranges([(250, 280)], 0, 120), [])

    def test_lists_only_segments_intersecting_manual_ranges(self):
        self.assertEqual(
            active_segment_indices([(615, 2527)], 2645, 120),
            list(range(5, 22)),
        )

    def test_segment_index_includes_partial_last_segment(self):
        self.assertEqual(active_segment_indices([(240, 250)], 250, 120), [2])


if __name__ == "__main__":
    unittest.main()
