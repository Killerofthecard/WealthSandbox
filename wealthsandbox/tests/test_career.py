"""Tests for the CareerSystem."""

import unittest
from wealthsandbox.systems.career import CareerSystem, DEFAULT_OCCUPATIONS, Occupation
from wealthsandbox.types import Action, AgentState, CareerMove, JobStatus
from wealthsandbox.config import TAX_RATE, UPSKILL_MONTHS


class TestCareerSystem(unittest.TestCase):

    def setUp(self):
        self.sys = CareerSystem(seed=0, layoff_base_rate=0.0)  # deterministic, no layoff in tests

    # --- Registry ---

    def test_default_occupations_loaded(self):
        self.assertIn("software_engineer", DEFAULT_OCCUPATIONS)
        self.assertIn("nurse", DEFAULT_OCCUPATIONS)
        self.assertEqual(len(DEFAULT_OCCUPATIONS), 7)

    def test_get_occupation(self):
        occ = self.sys.get_occupation("software_engineer")
        self.assertEqual(occ.industry, "tech")
        self.assertEqual(occ.base_monthly_salary, 8_800.0)
        self.assertEqual(occ.skill_sensitivity, 0.06)
        self.assertGreater(len(occ.tiers), 0)

    def test_unknown_occupation_raises(self):
        with self.assertRaises(ValueError):
            self.sys.get_occupation("astronaut")

    def test_list_occupations(self):
        ids = self.sys.list_occupations()
        self.assertEqual(len(ids), 7)

    # --- Income (general_skill affects salary) ---

    def test_monthly_base_salary_skill_effect(self):
        state_low = AgentState(
            occupation_id="software_engineer",
            general_skill=1,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
        )
        state_ref = AgentState(
            occupation_id="software_engineer",
            general_skill=3,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
        )
        state_high = AgentState(
            occupation_id="software_engineer",
            general_skill=5,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
        )

        low = self.sys.get_monthly_base_salary(state_low)
        ref = self.sys.get_monthly_base_salary(state_ref)
        high = self.sys.get_monthly_base_salary(state_high)

        self.assertAlmostEqual(ref, 8_800.0, delta=1)
        self.assertLess(low, ref)
        self.assertGreater(high, ref)

    def test_monthly_base_salary_no_occupation(self):
        state = AgentState(occupation_id="")
        self.assertEqual(self.sys.get_monthly_base_salary(state), 0.0)

    def test_compute_monthly_after_tax_income(self):
        state = AgentState(
            occupation_id="software_engineer",
            general_skill=3,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
        )
        self.sys.tick(state, {})
        income = self.sys.compute_monthly_after_tax_income(state)
        expected = 8800.0 * (1 - TAX_RATE)
        self.assertAlmostEqual(income, round(expected, 2), delta=0.1)

    def test_salary_scales_with_price_level(self):
        """Nominal salary grows with the CPI price level (cost-of-living adjustment)."""
        state = AgentState(
            occupation_id="software_engineer",
            general_skill=3,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
        )
        self.sys.tick(state, {"price_level": 1.0})
        base = self.sys.get_monthly_base_salary(state)
        self.sys.tick(state, {"price_level": 2.0})
        inflated = self.sys.get_monthly_base_salary(state)
        self.assertAlmostEqual(inflated, base * 2.0, places=4)

    # --- Tick (auto income) ---

    def test_tick_auto_income_when_employed(self):
        state = AgentState(
            occupation_id="software_engineer",
            general_skill=3,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"software_engineer": 1},
            cash=0.0,
        )
        self.sys.tick(state, {})
        self.assertGreater(state.cash, 0.0)
        self.assertGreater(state.monthly_after_tax_income, 0.0)

    def test_tick_no_income_when_unemployed(self):
        state = AgentState(
            occupation_id="software_engineer",
            general_skill=3,
            job_status=JobStatus.UNEMPLOYED,
            occupation_skills={"software_engineer": 1},
            cash=1000.0,
        )
        self.sys.tick(state, {})
        self.assertEqual(state.monthly_after_tax_income, 0.0)
        self.assertEqual(state.cash, 1000.0)

    def test_tick_no_income_no_occupation(self):
        state = AgentState(
            occupation_id="",
            job_status=JobStatus.UNEMPLOYED,
            cash=1000.0,
        )
        self.sys.tick(state, {})
        self.assertEqual(state.monthly_after_tax_income, 0.0)

    # --- Handle action ---

    def test_handle_action_switch_occupation(self):
        state = AgentState(
            occupation_id="",
            general_skill=5,
            job_status=JobStatus.UNEMPLOYED,
            cash=10_000.0,
        )
        action = Action(career_move=CareerMove.SWITCH_OCCUPATION, target_occupation_id="manufacturing_worker")
        consumed = self.sys.handle_action(action, state)
        self.assertTrue(consumed)
        self.assertEqual(state.occupation_id, "manufacturing_worker")

    def test_handle_action_upskill(self):
        state = AgentState(general_skill=3, cash=10_000.0)
        action = Action(career_move=CareerMove.UPSKILL)
        consumed = self.sys.handle_action(action, state)
        self.assertTrue(consumed)
        self.assertEqual(state.upskill_months_remaining, UPSKILL_MONTHS)

    def test_handle_action_quit_job(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 1},
        )
        action = Action(career_move=CareerMove.QUIT_JOB)
        consumed = self.sys.handle_action(action, state)
        self.assertTrue(consumed)
        self.assertEqual(state.job_status, JobStatus.UNEMPLOYED)
        self.assertEqual(state.occupation_id, "")

    def test_handle_action_intensive_work(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            general_skill=1,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 1},
        )
        action = Action(career_move=CareerMove.INTENSIVE_WORK)
        consumed = self.sys.handle_action(action, state)
        self.assertTrue(consumed)
        self.assertEqual(state.intensive_work_months_remaining, 3)

    # --- Quit job ---

    def test_quit_rejected_when_unemployed(self):
        state = AgentState(occupation_id="", job_status=JobStatus.UNEMPLOYED)
        self.sys.process_quit_job(state)
        self.assertTrue(any("Cannot quit" in e for e in state.last_month_events))

    # --- Energy tests (via EnergySystem) ---

    def test_energy_consumed_during_training(self):
        from wealthsandbox.systems.energy import EnergySystem
        es = EnergySystem()
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=3,
            energy=1.0,
            training_months_remaining=3,
            training_target_occupation="software_engineer",
            occupation_skills={"manufacturing_worker": 1},
        )
        es.tick(state, {})
        self.assertLess(state.energy, 1.0)

    def test_energy_recovered_when_not_training(self):
        from wealthsandbox.systems.energy import EnergySystem
        es = EnergySystem()
        state = AgentState(
            energy=0.5,
            training_months_remaining=0,
        )
        es.tick(state, {})
        self.assertGreater(state.energy, 0.5)

    # --- Process switch ---

    def test_process_switch_occupation_immediate(self):
        state = AgentState(
            occupation_id="",
            general_skill=5,
            job_status=JobStatus.UNEMPLOYED,
            cash=10_000.0,
        )
        self.sys.process_switch_occupation(state, "manufacturing_worker")
        self.assertEqual(state.occupation_id, "manufacturing_worker")
        self.assertEqual(state.job_status, JobStatus.EMPLOYED)
        self.assertEqual(state.cash, 10_000.0)  # safety-net job — always free
        self.assertEqual(state.general_skill, 5)  # first job: 100% retention

    def test_process_switch_occupation_starts_training(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            general_skill=5,
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 1},
            cash=20_000.0,
        )
        self.sys.process_switch_occupation(state, "software_engineer")
        self.assertEqual(state.occupation_id, "manufacturing_worker")  # unchanged
        self.assertEqual(state.training_months_remaining, 4)
        self.assertEqual(state.training_target_occupation, "software_engineer")

    # --- Process upskill (general skill) ---

    def test_process_upskill(self):
        state = AgentState(general_skill=3, cash=10_000.0)
        self.sys.process_upskill(state)
        self.assertEqual(state.upskill_months_remaining, UPSKILL_MONTHS)
        self.assertLess(state.cash, 10_000.0)

    def test_process_upskill_at_max_skill(self):
        state = AgentState(general_skill=10, cash=20_000.0)
        self.sys.process_upskill(state)
        self.assertEqual(state.upskill_months_remaining, 0)
        self.assertTrue(any("Already at maximum" in e for e in state.last_month_events))

    def test_tick_upskill_completion(self):
        state = AgentState(general_skill=3, upskill_months_remaining=1)
        self.sys.tick_upskill(state)
        self.assertEqual(state.upskill_months_remaining, 0)
        self.assertEqual(state.general_skill, 4)

    # --- Entry barriers ---

    def test_min_skill_requirement_rejected(self):
        state = AgentState(
            occupation_id="",
            general_skill=1,
            cash=50_000.0,
        )
        self.sys.process_switch_occupation(state, "software_engineer")
        self.assertEqual(state.occupation_id, "")
        self.assertTrue(any("General skill too low" in e for e in state.last_month_events))

    def test_insufficient_cash_for_entry(self):
        state = AgentState(
            occupation_id="",
            general_skill=5,
            cash=100.0,
        )
        self.sys.process_switch_occupation(state, "software_engineer")
        self.assertEqual(state.occupation_id, "")
        self.assertTrue(any("Cannot afford" in e for e in state.last_month_events))

    def test_cannot_switch_during_training(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            general_skill=5,
            cash=50_000.0,
            training_months_remaining=3,
            training_target_occupation="software_engineer",
            occupation_skills={"manufacturing_worker": 1},
        )
        self.sys.process_switch_occupation(state, "nurse")
        self.assertTrue(any("already training" in e.lower() for e in state.last_month_events))

    # --- Training tick ---

    def test_training_completion_applies_skill_retention(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            general_skill=5,
            job_status=JobStatus.EMPLOYED,
            training_months_remaining=1,
            training_target_occupation="software_engineer",
            occupation_skills={"manufacturing_worker": 3},
        )
        self.sys.tick_training(state)
        self.assertEqual(state.training_months_remaining, 0)
        self.assertEqual(state.occupation_id, "software_engineer")
        # Cross-industry: general_skill 5 * 0.2 = 1
        self.assertEqual(state.general_skill, 1)
        # Occupation skill resets to tier 0's min
        self.assertEqual(state.occupation_skills.get("software_engineer", 0), 1)

    # --- Skill retention ---

    def test_skill_retention_same_industry(self):
        retention = self.sys.get_skill_retention("software_engineer", "data_scientist")
        self.assertAlmostEqual(retention, 0.8)

    def test_skill_retention_cross_industry(self):
        retention = self.sys.get_skill_retention("manufacturing_worker", "software_engineer")
        self.assertAlmostEqual(retention, 0.2)

    def test_skill_retention_from_unemployed(self):
        retention = self.sys.get_skill_retention("", "software_engineer")
        self.assertAlmostEqual(retention, 1.0)

    # --- Occupation details ---

    def test_occupation_details_include_new_fields(self):
        details = self.sys.get_occupation_details()
        sw = details["software_engineer"]
        self.assertEqual(sw["min_general_skill"], 4)
        self.assertEqual(sw["entry_cost"], 10_000.0)
        self.assertEqual(sw["training_months"], 4)
        self.assertGreater(len(sw["tiers"]), 0)

    def test_manufacturing_worker_no_barriers(self):
        details = self.sys.get_occupation_details()
        mw = details["manufacturing_worker"]
        self.assertEqual(mw["min_general_skill"], 1)
        self.assertEqual(mw["entry_cost"], 0)
        self.assertEqual(mw["training_months"], 0)

    # --- min_health ---

    def test_occupation_min_health_present(self):
        details = self.sys.get_occupation_details()
        for occ_id, detail in details.items():
            self.assertIn("min_health", detail)
            self.assertGreaterEqual(detail["min_health"], 0.0)

    # --- Intensive work ---

    def test_intensive_work_rejected_when_unemployed(self):
        state = AgentState(
            occupation_id="",
            job_status=JobStatus.UNEMPLOYED,
        )
        self.sys.process_intensive_work(state)
        self.assertTrue(any("Must be employed" in e for e in state.last_month_events))

    def test_tick_intensive_work_completion(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 2},
            intensive_work_months_remaining=1,
        )
        self.sys.tick_intensive_work(state)
        self.assertEqual(state.intensive_work_months_remaining, 0)
        self.assertEqual(state.occupation_skills["manufacturing_worker"], 3)

    # --- Tenure and passive occ_skill ---

    def test_tenure_increments_each_month(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 1},
            tenure_months=0,
        )
        self.sys.tick(state, {})
        self.assertEqual(state.tenure_months, 1)

    # --- Tier promotion ---

    def test_tier_promotion_when_conditions_met(self):
        sys = CareerSystem(occ_skill_passive_months=12)
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=1,
            occupation_skills={"manufacturing_worker": 3},
            tenure_months=12,
            current_tier=0,
        )
        sys.tick(state, {})
        self.assertEqual(state.current_tier, 1)  # promoted to Skilled

    # --- Health forced resignation ---

    def test_forced_resign_when_health_below_minimum(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=1,
            occupation_skills={"manufacturing_worker": 3},
            health=0.3,  # below min_health=0.6
            cash=10_000.0,
        )
        self.sys.tick(state, {})
        self.assertEqual(state.job_status, JobStatus.UNEMPLOYED)
        self.assertEqual(state.occupation_id, "")
        self.assertTrue(any("Forced to resign" in e for e in state.last_month_events))


class TestHealthSystem(unittest.TestCase):

    def setUp(self):
        from wealthsandbox.systems.health import HealthSystem
        self.hs = HealthSystem()

    def _state(self, age: int, health: float = 1.0) -> AgentState:
        return AgentState(age=age, health=health)

    def test_decline_rate_20s(self):
        s = self._state(25)
        self.hs.finalize(s, {})
        self.assertAlmostEqual(s.health, 1.0 - 0.0003)

    def test_decline_rate_30s(self):
        s = self._state(35)
        self.hs.finalize(s, {})
        self.assertAlmostEqual(s.health, 1.0 - 0.002)

    def test_decline_rate_40s(self):
        s = self._state(45)
        self.hs.finalize(s, {})
        self.assertAlmostEqual(s.health, 1.0 - 0.006)

    def test_decline_rate_50_plus(self):
        s = self._state(55)
        self.hs.finalize(s, {})
        self.assertAlmostEqual(s.health, 1.0 - 0.012)

    def test_death_when_health_zero(self):
        s = self._state(30, health=0.0001)
        self.hs.finalize(s, {})
        self.assertLessEqual(s.health, 0.0)
        self.assertEqual(self.hs.check_dead(s), "death")

    def test_no_death_when_health_positive(self):
        s = self._state(30, health=0.5)
        self.assertIsNone(self.hs.check_dead(s))


class TestEnergySystem(unittest.TestCase):

    def setUp(self):
        from wealthsandbox.systems.energy import EnergySystem
        self.es = EnergySystem(cost_per_upskill=0.4)

    def _state(self, energy: float = 1.0, training: int = 0) -> AgentState:
        return AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            energy=energy,
            training_months_remaining=training,
            occupation_skills={"manufacturing_worker": 1},
        )

    def test_upskill_deducts_energy(self):
        s = self._state(energy=0.8)
        action = Action(career_move=CareerMove.UPSKILL)
        self.es.handle_action(action, s)
        self.assertAlmostEqual(s.energy, 0.4)

    def test_intensive_work_deducts_energy(self):
        s = self._state(energy=0.8)
        action = Action(career_move=CareerMove.INTENSIVE_WORK)
        self.es.handle_action(action, s)
        self.assertAlmostEqual(s.energy, 0.3)

    def test_energy_drained_during_training(self):
        s = self._state(energy=0.5, training=3)
        self.es.tick(s, {})
        self.assertAlmostEqual(s.energy, 0.35)

    def test_energy_recovered_when_not_training(self):
        s = self._state(energy=0.5, training=0)
        self.es.tick(s, {})
        self.assertAlmostEqual(s.energy, 0.60)


if __name__ == "__main__":
    unittest.main()
