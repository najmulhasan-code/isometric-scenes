import unittest

from svg_scoring.motion import FRAME_COUNT, LAST_TIME, frame_times, judge_frames


def measures(changed: float, brightness: tuple[float, ...] = (200, 200, 200, 200),
             shares: tuple[float, ...] = (0.7, 0.2, 0.1)) -> dict:
    return {"colour_shares": list(shares), "brightness": list(brightness),
            "changed": [[0, 1, 0.0], [0, 2, changed], [1, 2, changed]]}


class FrameTimesTest(unittest.TestCase):
    def test_times_are_seeded_by_the_answer(self):
        self.assertEqual(frame_times("a"), frame_times("a"))
        self.assertNotEqual(frame_times("a"), frame_times("b"))

    def test_first_frame_is_at_zero_and_the_rest_in_range(self):
        times = frame_times("<svg/>")
        self.assertEqual(len(times), FRAME_COUNT)
        self.assertEqual(times[0], 0.0)
        self.assertTrue(all(0 < t <= LAST_TIME for t in times[1:]))


class JudgeFramesTest(unittest.TestCase):
    def test_a_small_local_change_moves(self):
        self.assertTrue(judge_frames(measures(0.01)).moves)

    def test_no_change_does_not_move(self):
        self.assertFalse(judge_frames(measures(0.0)).moves)

    def test_a_full_frame_change_does_not_move(self):
        self.assertFalse(judge_frames(measures(0.8)).moves)

    def test_a_single_change_at_the_start_does_not_move(self):
        frames = measures(0.0)
        frames["changed"] = [[0, 1, 0.01], [0, 2, 0.01], [0, 3, 0.01], [1, 2, 0.0], [1, 3, 0.0], [2, 3, 0.0]]
        self.assertFalse(judge_frames(frames).moves)

    def test_a_change_between_later_frames_moves(self):
        frames = measures(0.0)
        frames["changed"] = [[0, 1, 0.01], [0, 2, 0.0], [1, 2, 0.01]]
        self.assertTrue(judge_frames(frames).moves)

    def test_a_brightness_flash_does_not_move(self):
        self.assertFalse(judge_frames(measures(0.3, brightness=(40, 220, 40, 220))).moves)

    def test_one_colour_covering_almost_everything_is_blank(self):
        self.assertFalse(judge_frames(measures(0.01, shares=(1.0,))).not_blank)
        self.assertFalse(judge_frames(measures(0.01, shares=(0.99, 0.01))).not_blank)
        self.assertTrue(judge_frames(measures(0.01, shares=(0.88, 0.11, 0.005))).not_blank)


if __name__ == "__main__":
    unittest.main()
