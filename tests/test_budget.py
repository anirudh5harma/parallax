import threading
import time
import unittest

from deep_research.domain.budget import BudgetConfig, BudgetExceeded, BudgetManager


class BudgetManagerTests(unittest.TestCase):
    def test_primary_work_cannot_consume_followup_reserves(self) -> None:
        config = BudgetConfig.fast()
        budget = BudgetManager(config)
        for _ in range(config.max_pages - config.followup_page_reserve):
            budget.reserve_page(primary=True)
        for _ in range(config.max_sources - config.followup_source_reserve):
            budget.reserve_source(primary=True)

        with self.assertRaises(BudgetExceeded):
            budget.reserve_page(primary=True)
        with self.assertRaises(BudgetExceeded):
            budget.reserve_source(primary=True)

        budget.reserve_page()
        budget.reserve_source()
        self.assertEqual(
            config.max_pages - config.followup_page_reserve + 1,
            budget.snapshot().pages,
        )
        self.assertEqual(
            config.max_sources - config.followup_source_reserve + 1,
            budget.snapshot().sources,
        )

    def test_interactive_profiles_preserve_breadth_with_distinct_read_budgets(self) -> None:
        fast = BudgetConfig.fast()
        deep = BudgetConfig.deep()

        self.assertEqual((600, 220, 100, 35, 1, 900), (
            fast.max_sources,
            fast.max_pages,
            fast.followup_source_reserve,
            fast.followup_page_reserve,
            fast.max_followup_tasks,
            fast.wall_clock_timeout_seconds,
        ))
        self.assertEqual((800, 400, 120, 60, 2, 1200), (
            deep.max_sources,
            deep.max_pages,
            deep.followup_source_reserve,
            deep.followup_page_reserve,
            deep.max_followup_tasks,
            deep.wall_clock_timeout_seconds,
        ))

    def test_followup_reserves_must_fit_global_budgets(self) -> None:
        with self.assertRaisesRegex(ValueError, "followup_source_reserve"):
            BudgetConfig(max_sources=3, followup_source_reserve=4)
        with self.assertRaisesRegex(ValueError, "followup_page_reserve"):
            BudgetConfig(max_pages=3, followup_page_reserve=4)

    def test_source_screening_ceiling_is_atomic(self) -> None:
        budget = BudgetManager(BudgetConfig(max_sources=3))
        outcomes: list[bool] = []

        def reserve() -> None:
            try:
                budget.reserve_source()
                outcomes.append(True)
            except BudgetExceeded:
                outcomes.append(False)

        threads = [threading.Thread(target=reserve) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(3, sum(outcomes))
        self.assertEqual(3, budget.snapshot().sources)

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
