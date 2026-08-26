import threading
import time
import unittest

from deep_research.domain.budget import BudgetConfig, BudgetExceeded, BudgetManager


class BudgetManagerTests(unittest.TestCase):
    def test_search_ceiling_is_atomic(self) -> None:
        budget = BudgetManager(BudgetConfig(max_searches=3))
        outcomes: list[bool] = []

        def reserve() -> None:
            try:
                budget.reserve_search()
                outcomes.append(True)
            except BudgetExceeded:
                outcomes.append(False)

        threads = [threading.Thread(target=reserve) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(3, sum(outcomes))
        self.assertEqual(3, budget.snapshot().searches)

    def test_deadline_is_enforced(self) -> None:
        budget = BudgetManager(BudgetConfig(wall_clock_timeout_seconds=0.01))
        time.sleep(0.02)
        with self.assertRaises(BudgetExceeded):
            budget.check_time()

    def test_rejects_invalid_hard_limits(self) -> None:
        with self.assertRaises(ValueError):
            BudgetConfig(max_research_tasks=7)


if __name__ == "__main__":
    unittest.main()
