import os
import unittest
from unittest import mock

from scripts import qwen_swehero_train as train


class QwenSweHeroTrainPlanTests(unittest.TestCase):
    def test_default_short_run_keeps_paper_batch_and_epoch_count(self):
        args = train.parse_args([])
        plan = train.build_training_plan(
            num_unique_examples=args.num_examples,
            global_batch_size=args.global_batch_size,
            per_device_train_batch_size=args.per_device_train_batch_size,
            world_size=args.world_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
        )

        self.assertEqual(args.model_context_length, train.smoke.PAPER_CONTEXT_LENGTH)
        self.assertEqual(args.max_length, train.DEFAULT_TRAIN_MAX_LENGTH)
        self.assertEqual(args.train_mode, "lora")
        self.assertFalse(args.enable_wandb)
        self.assertEqual(plan.batch.global_batch_size, 32)
        self.assertEqual(plan.batch.effective_global_batch_size, 32)
        self.assertEqual(plan.batch.gradient_accumulation_steps, 32)
        self.assertEqual(plan.items_per_epoch, 32)
        self.assertEqual(plan.steps_per_epoch, 1)
        self.assertEqual(plan.total_optimizer_steps, 3)

    def test_plan_pads_partial_epoch_to_effective_global_batch(self):
        plan = train.build_training_plan(
            num_unique_examples=33,
            global_batch_size=32,
            per_device_train_batch_size=1,
            world_size=1,
            gradient_accumulation_steps=None,
            num_train_epochs=3,
            max_steps=0,
        )

        self.assertEqual(plan.items_per_epoch, 64)
        self.assertEqual(plan.steps_per_epoch, 2)
        self.assertEqual(plan.total_optimizer_steps, 6)

    def test_max_steps_override_extends_epoch_length(self):
        plan = train.build_training_plan(
            num_unique_examples=8,
            global_batch_size=32,
            per_device_train_batch_size=1,
            world_size=1,
            gradient_accumulation_steps=None,
            num_train_epochs=3,
            max_steps=4,
        )

        self.assertEqual(plan.items_per_epoch, 128)
        self.assertEqual(plan.steps_per_epoch, 4)
        self.assertEqual(plan.total_optimizer_steps, 4)
        self.assertEqual(plan.max_steps_override, 4)

    def test_paper_alignment_declares_short_run_deviations(self):
        args = train.parse_args([])
        alignment = train._paper_alignment(args)

        self.assertEqual(
            alignment["kept"]["model_context_length"],
            train.smoke.PAPER_CONTEXT_LENGTH,
        )
        self.assertIn(
            "7B direct-to-hero is a scale-study extension",
            alignment["intentional_deviations"][0],
        )
        self.assertTrue(
            any(
                "LoRA" in deviation or "lora" in deviation
                for deviation in alignment["intentional_deviations"]
            )
        )

    def test_torchrun_rank_env_drives_cuda_device_selection(self):
        class FakeCuda:
            def __init__(self):
                self.selected_device = None

            def is_available(self):
                return True

            def set_device(self, local_rank):
                self.selected_device = local_rank

        class FakeTorch:
            cuda = FakeCuda()

        with mock.patch.dict(os.environ, {"LOCAL_RANK": "6"}, clear=False):
            device = train._select_cuda_device(FakeTorch)

        self.assertEqual(device, "cuda:6")
        self.assertEqual(FakeTorch.cuda.selected_device, 6)

    def test_main_process_detection_uses_global_rank(self):
        with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False):
            self.assertTrue(train._is_main_process_from_env())

        with mock.patch.dict(os.environ, {"RANK": "3"}, clear=False):
            self.assertFalse(train._is_main_process_from_env())

    def test_base_causal_lm_unwraps_ddp_before_peft(self):
        class PeftLikeModel:
            def get_base_model(self):
                return "base"

        class DdpLikeWrapper:
            module = PeftLikeModel()

        self.assertEqual(train._base_causal_lm(DdpLikeWrapper()), "base")


if __name__ == "__main__":
    unittest.main()
