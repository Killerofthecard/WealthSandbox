"""Tests for the centralized ActionValidator and guard functions."""

import unittest

from wealthsandbox.systems.career import CareerSystem
from wealthsandbox.types import Action, AgentState, CareerMove, JobStatus
from wealthsandbox.validator import (
    ActionValidator,
    GuardResult,
    guard_switch_occupation,
    guard_switch_occupation_target,
    guard_upskill,
    guard_intensive_work,
    guard_quit_job,
    guard_deposit_amount,
    guard_withdraw_amount,
    guard_borrow_amount,
    guard_repay_amount,
    guard_rest,
)


class TestGuardResult(unittest.TestCase):

    def test_ok(self):
        r = GuardResult.ok()
        self.assertTrue(r.allowed)
        self.assertEqual(r.event_key, "")

    def test_reject(self):
        r = GuardResult.reject("key", "message")
        self.assertFalse(r.allowed)
        self.assertEqual(r.event_key, "key")
        self.assertEqual(r.message, "message")


class TestGuardFunctions(unittest.TestCase):

    def setUp(self):
        self.career = CareerSystem()

    # --- guard_switch_occupation ---

    def test_switch_blocked_during_training(self):
        state = AgentState(
            training_months_remaining=3,
            training_target_occupation="software_engineer",
        )
        r = guard_switch_occupation(state, self.career)
        self.assertFalse(r.allowed)

    def test_switch_allowed_when_not_training(self):
        state = AgentState(training_months_remaining=0)
        r = guard_switch_occupation(state, self.career)
        self.assertTrue(r.allowed)

    # --- guard_switch_occupation_target ---

    def test_switch_target_empty_id(self):
        state = AgentState()
        r = guard_switch_occupation_target(state, self.career, "")
        self.assertFalse(r.allowed)

    def test_switch_target_invalid_occupation(self):
        state = AgentState()
        r = guard_switch_occupation_target(state, self.career, "astronaut")
        self.assertFalse(r.allowed)

    def test_switch_target_skill_too_low(self):
        state = AgentState(general_skill=1, cash=50_000)
        r = guard_switch_occupation_target(state, self.career, "software_engineer")
        self.assertFalse(r.allowed)
        self.assertIn("general skill", r.message.lower())

    def test_switch_target_insufficient_cash(self):
        state = AgentState(general_skill=5, cash=100)
        r = guard_switch_occupation_target(state, self.career, "software_engineer")
        self.assertFalse(r.allowed)
        self.assertIn("insufficient cash", r.message.lower())

    def test_switch_target_allowed(self):
        state = AgentState(general_skill=5, cash=50_000)
        r = guard_switch_occupation_target(state, self.career, "software_engineer")
        self.assertTrue(r.allowed)

    def test_switch_target_no_barrier_occupation(self):
        state = AgentState(general_skill=1, cash=5_000)
        r = guard_switch_occupation_target(state, self.career, "manufacturing_worker")
        self.assertTrue(r.allowed)

    # --- guard_upskill ---

    def test_upskill_at_max_skill(self):
        state = AgentState(general_skill=10)
        r = guard_upskill(state, self.career)
        self.assertFalse(r.allowed)

    def test_upskill_already_in_progress(self):
        state = AgentState(general_skill=3, upskill_months_remaining=4)
        r = guard_upskill(state, self.career)
        self.assertFalse(r.allowed)

    def test_upskill_insufficient_cash(self):
        state = AgentState(general_skill=3, cash=100)
        r = guard_upskill(state, self.career)
        self.assertFalse(r.allowed)

    def test_upskill_blocked_during_training(self):
        state = AgentState(
            general_skill=3,
            cash=10_000,
            training_months_remaining=3,
            training_target_occupation="software_engineer",
        )
        r = guard_upskill(state, self.career)
        self.assertFalse(r.allowed)
        self.assertIn("training", r.message.lower())

    def test_upskill_allowed(self):
        state = AgentState(general_skill=3, cash=10_000)
        r = guard_upskill(state, self.career)
        self.assertTrue(r.allowed)

    # --- guard_intensive_work ---

    def test_intensive_work_not_employed(self):
        state = AgentState(job_status=JobStatus.UNEMPLOYED)
        r = guard_intensive_work(state, self.career)
        self.assertFalse(r.allowed)

    def test_intensive_work_allowed(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            occupation_skills={"manufacturing_worker": 1},
        )
        r = guard_intensive_work(state, self.career)
        self.assertTrue(r.allowed)

    # --- Energy guard (via closure) ---

    def test_upskill_blocked_by_low_energy(self):
        validator = ActionValidator(self.career, energy_threshold=0.4)
        state = AgentState(general_skill=3, cash=10_000, energy=0.2)
        action = Action(career_move=CareerMove.UPSKILL)
        r = validator.validate(action, state)
        self.assertFalse(r.allowed)
        self.assertIn("energy", r.message.lower())

    def test_upskill_allowed_with_sufficient_energy(self):
        validator = ActionValidator(self.career, energy_threshold=0.4)
        state = AgentState(general_skill=3, cash=10_000, energy=0.6)
        action = Action(career_move=CareerMove.UPSKILL)
        r = validator.validate(action, state)
        self.assertTrue(r.allowed)

    # --- Health guard on switch ---

    def test_switch_blocked_by_low_health(self):
        state = AgentState(general_skill=5, cash=50_000, health=0.4)
        r = guard_switch_occupation_target(state, self.career, "manufacturing_worker")
        self.assertFalse(r.allowed)
        self.assertIn("health", r.message.lower())

    def test_switch_allowed_with_sufficient_health(self):
        state = AgentState(general_skill=1, cash=5_000, health=0.8)
        r = guard_switch_occupation_target(state, self.career, "manufacturing_worker")
        self.assertTrue(r.allowed)

    def test_switch_desk_job_allowed_with_low_health(self):
        state = AgentState(general_skill=5, cash=50_000, health=0.4)
        r = guard_switch_occupation_target(state, self.career, "civil_servant")
        self.assertTrue(r.allowed)

    # --- guard_quit_job ---

    def test_quit_when_employed(self):
        state = AgentState(job_status=JobStatus.EMPLOYED)
        r = guard_quit_job(state, self.career)
        self.assertTrue(r.allowed)

    def test_quit_when_unemployed(self):
        state = AgentState(job_status=JobStatus.UNEMPLOYED)
        r = guard_quit_job(state, self.career)
        self.assertFalse(r.allowed)

    # --- guard_rest ---

    def test_rest_rejected_when_fully_rested(self):
        state = AgentState(health=1.0, energy=1.0)
        r = guard_rest(state, self.career)
        self.assertFalse(r.allowed)

    def test_rest_allowed_when_low_health(self):
        state = AgentState(health=0.5, energy=1.0)
        r = guard_rest(state, self.career)
        self.assertTrue(r.allowed)

    def test_rest_allowed_when_low_energy(self):
        state = AgentState(health=1.0, energy=0.3)
        r = guard_rest(state, self.career)
        self.assertTrue(r.allowed)

    # --- guard_deposit_amount ---

    def test_deposit_amount_zero_rejected(self):
        state = AgentState(cash=5_000)
        r = guard_deposit_amount(state, self.career, 0)
        self.assertFalse(r.allowed)
        self.assertIn("greater than zero", r.message.lower())

    def test_deposit_amount_exceeds_cash_rejected(self):
        state = AgentState(cash=5_000)
        r = guard_deposit_amount(state, self.career, 6_000)
        self.assertFalse(r.allowed)

    def test_deposit_amount_buffer_violation_rejected(self):
        state = AgentState(cash=5_000)
        r = guard_deposit_amount(state, self.career, 4_000)
        self.assertFalse(r.allowed)
        self.assertIn("living expenses", r.message.lower())

    def test_deposit_amount_allowed(self):
        state = AgentState(cash=5_000)
        r = guard_deposit_amount(state, self.career, 2_000)
        self.assertTrue(r.allowed)

    # --- guard_withdraw_amount ---

    def test_withdraw_amount_zero_rejected(self):
        state = AgentState(savings=3_000)
        r = guard_withdraw_amount(state, self.career, 0)
        self.assertFalse(r.allowed)
        self.assertIn("greater than zero", r.message.lower())

    def test_withdraw_amount_exceeds_savings_rejected(self):
        state = AgentState(savings=3_000)
        r = guard_withdraw_amount(state, self.career, 5_000)
        self.assertFalse(r.allowed)

    def test_withdraw_amount_allowed(self):
        state = AgentState(savings=3_000)
        r = guard_withdraw_amount(state, self.career, 2_000)
        self.assertTrue(r.allowed)

    # --- guard_borrow_amount ---

    def test_borrow_amount_zero_rejected(self):
        state = AgentState(loan_balance=0)
        r = guard_borrow_amount(state, self.career, 0)
        self.assertFalse(r.allowed)
        self.assertIn("greater than zero", r.message.lower())

    def test_borrow_amount_exceeds_limit_rejected(self):
        state = AgentState(
            loan_balance=7_000,
            job_status=JobStatus.UNEMPLOYED,
        )
        r = guard_borrow_amount(state, self.career, 2_000)
        self.assertFalse(r.allowed)

    def test_borrow_amount_allowed(self):
        state = AgentState(loan_balance=0, job_status=JobStatus.UNEMPLOYED)
        r = guard_borrow_amount(state, self.career, 5_000)
        self.assertTrue(r.allowed)

    # --- guard_repay_amount ---

    def test_repay_amount_zero_rejected(self):
        state = AgentState(cash=5_000, loan_balance=3_000)
        r = guard_repay_amount(state, self.career, 0)
        self.assertFalse(r.allowed)
        self.assertIn("greater than zero", r.message.lower())

    def test_repay_amount_exceeds_cash_rejected(self):
        state = AgentState(cash=1_000, loan_balance=3_000)
        r = guard_repay_amount(state, self.career, 2_000)
        self.assertFalse(r.allowed)

    def test_repay_amount_exceeds_loan_rejected(self):
        state = AgentState(cash=5_000, loan_balance=2_000)
        r = guard_repay_amount(state, self.career, 3_000)
        self.assertFalse(r.allowed)

    def test_repay_amount_allowed(self):
        state = AgentState(cash=5_000, loan_balance=3_000)
        r = guard_repay_amount(state, self.career, 2_000)
        self.assertTrue(r.allowed)


class TestActionValidator(unittest.TestCase):

    def setUp(self):
        self.career = CareerSystem()
        self.validator = ActionValidator(self.career)

    def test_validate_upskill_allowed(self):
        state = AgentState(general_skill=3, cash=10_000)
        action = Action(career_move=CareerMove.UPSKILL)
        r = self.validator.validate(action, state)
        self.assertTrue(r.allowed)

    def test_validate_upskill_blocked(self):
        state = AgentState(general_skill=10)
        action = Action(career_move=CareerMove.UPSKILL)
        r = self.validator.validate(action, state)
        self.assertFalse(r.allowed)

    def test_validate_quit_allowed(self):
        state = AgentState(job_status=JobStatus.EMPLOYED)
        action = Action(career_move=CareerMove.QUIT_JOB)
        r = self.validator.validate(action, state)
        self.assertTrue(r.allowed)

    def test_validate_quit_blocked(self):
        state = AgentState(job_status=JobStatus.UNEMPLOYED)
        action = Action(career_move=CareerMove.QUIT_JOB)
        r = self.validator.validate(action, state)
        self.assertFalse(r.allowed)

    def test_validate_none_always_ok(self):
        state = AgentState()
        action = Action(career_move=CareerMove.NONE)
        r = self.validator.validate(action, state)
        self.assertTrue(r.allowed)

    # --- available_actions ---

    def test_available_actions_employed(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=3,
            cash=10_000,
            occupation_skills={"manufacturing_worker": 1},
        )
        avail = self.validator.available_actions(state)
        self.assertTrue(avail["upskill"]["allowed"])
        self.assertTrue(avail["quit_job"]["allowed"])
        self.assertTrue(avail["switch_occupation"]["allowed"])
        self.assertTrue(avail["intensive_work"]["allowed"])

    def test_available_actions_during_training(self):
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=3,
            cash=10_000,
            training_months_remaining=3,
            training_target_occupation="software_engineer",
            occupation_skills={"manufacturing_worker": 1},
        )
        avail = self.validator.available_actions(state)
        self.assertFalse(avail["upskill"]["allowed"])
        self.assertTrue(avail["quit_job"]["allowed"])
        self.assertFalse(avail["switch_occupation"]["allowed"])

    def test_available_actions_unemployed(self):
        state = AgentState(
            job_status=JobStatus.UNEMPLOYED,
            general_skill=1,
            cash=100,
        )
        avail = self.validator.available_actions(state)
        self.assertFalse(avail["upskill"]["allowed"])
        self.assertFalse(avail["quit_job"]["allowed"])
        self.assertFalse(avail["intensive_work"]["allowed"])
        self.assertTrue(avail["switch_occupation"]["allowed"])

    def test_available_actions_low_energy_shows_reason(self):
        validator = ActionValidator(self.career, energy_threshold=0.4)
        state = AgentState(
            occupation_id="manufacturing_worker",
            job_status=JobStatus.EMPLOYED,
            general_skill=3,
            cash=10_000,
            energy=0.2,
            occupation_skills={"manufacturing_worker": 1},
        )
        avail = validator.available_actions(state)
        self.assertFalse(avail["upskill"]["allowed"])
        self.assertIn("energy", avail["upskill"]["reason"].lower())

    # --- rest / medical_care guards (via validator) ---

    def test_rest_allowed_via_validator(self):
        state = AgentState(health=0.5)
        action = Action(career_move=CareerMove.REST)
        r = self.validator.validate(action, state)
        self.assertTrue(r.allowed)

    def test_medical_care_allowed_with_cash(self):
        state = AgentState(cash=5_000, health=0.5)
        action = Action(career_move=CareerMove.MEDICAL_CARE)
        r = self.validator.validate(action, state)
        self.assertTrue(r.allowed)

    def test_medical_care_rejected_insufficient_cash(self):
        state = AgentState(cash=1_000, health=0.5)
        action = Action(career_move=CareerMove.MEDICAL_CARE)
        r = self.validator.validate(action, state)
        self.assertFalse(r.allowed)
        self.assertIn("cash", r.message.lower())

    def test_medical_care_rejected_at_yearly_limit(self):
        state = AgentState(cash=50_000, health=0.5, medical_care_uses_this_year=2)
        action = Action(career_move=CareerMove.MEDICAL_CARE)
        r = self.validator.validate(action, state)
        self.assertFalse(r.allowed)
        self.assertIn("times", r.message.lower())

    # --- Bank action-specific validators ---

    def test_validate_deposit_rejects_amount_zero(self):
        state = AgentState(cash=5_000)
        action = Action(career_move=CareerMove.DEPOSIT, amount=0)
        r = self.validator.validate_deposit(action, state)
        self.assertFalse(r.allowed)

    def test_validate_deposit_allows_valid_amount(self):
        state = AgentState(cash=5_000)
        action = Action(career_move=CareerMove.DEPOSIT, amount=3_000)
        r = self.validator.validate_deposit(action, state)
        self.assertTrue(r.allowed)

    def test_validate_withdraw_rejects_amount_zero(self):
        state = AgentState(savings=3_000)
        action = Action(career_move=CareerMove.WITHDRAW, amount=0)
        r = self.validator.validate_withdraw(action, state)
        self.assertFalse(r.allowed)

    def test_validate_withdraw_allows_valid_amount(self):
        state = AgentState(savings=3_000)
        action = Action(career_move=CareerMove.WITHDRAW, amount=2_000)
        r = self.validator.validate_withdraw(action, state)
        self.assertTrue(r.allowed)

    def test_validate_borrow_rejects_amount_zero(self):
        state = AgentState(job_status=JobStatus.UNEMPLOYED)
        action = Action(career_move=CareerMove.BORROW, amount=0)
        r = self.validator.validate_borrow(action, state)
        self.assertFalse(r.allowed)

    def test_validate_borrow_allows_valid_amount(self):
        state = AgentState(job_status=JobStatus.UNEMPLOYED)
        action = Action(career_move=CareerMove.BORROW, amount=5_000)
        r = self.validator.validate_borrow(action, state)
        self.assertTrue(r.allowed)

    def test_validate_repay_rejects_amount_zero(self):
        state = AgentState(cash=5_000, loan_balance=3_000)
        action = Action(career_move=CareerMove.REPAY, amount=0)
        r = self.validator.validate_repay(action, state)
        self.assertFalse(r.allowed)

    def test_validate_repay_allows_valid_amount(self):
        state = AgentState(cash=5_000, loan_balance=3_000)
        action = Action(career_move=CareerMove.REPAY, amount=2_000)
        r = self.validator.validate_repay(action, state)
        self.assertTrue(r.allowed)


if __name__ == "__main__":
    unittest.main()
