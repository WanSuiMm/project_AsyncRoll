import copy
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from asyncroll import experiment


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "experiments" / "single_5090.json"


def _test_inputs(root: Path) -> dict:
    return {
        "model_path": root / "arbitrary-model-folder",
        "workload_path": root / "selected.jsonl",
        "tool_root": root / "functions",
        "model_snapshot": {"weights_revision_verified": True},
        "workload_sha256": "workload-digest",
        "workload_ids": ["one", "two"],
        "tool_hashes": {"add:functions/add.py": "tool-digest"},
        "declared_workload_revision_status": "declared_source_revision_not_independently_verified",
    }


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.config = experiment.load_config(CONFIG_PATH)

    def test_current_config_and_ordered_seed_pairs(self):
        experiment.validate_config(self.config)
        pairs = experiment.build_pairs(self.config)
        base = self.config["seed"]["base"]
        self.assertEqual([pair["seed"] for pair in pairs], [base, base + 1, base + 2])
        self.assertEqual([pair["order"] for pair in pairs],
                         self.config["comparison"]["orders"])
        self.assertEqual([pair["pair_id"] for pair in pairs],
                         ["pair-01", "pair-02", "pair-03"])
        self.assertEqual(len({pair["seed"] for pair in pairs}), 3)

        drifted = copy.deepcopy(self.config)
        drifted["comparison"]["orders"][1] = ["fifo", "asyncroll"]
        with self.assertRaises(experiment.ExperimentError):
            experiment.validate_config(drifted)

    def test_formal_client_is_unprofiled_and_uses_physical_nvml_index(self):
        self.config["server"]["port"] = 8012
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _test_inputs(root)
            inputs["workload_path"].touch()
            inputs["tool_root"].mkdir()
            cmd = experiment.client_command(
                self.config, inputs, policy="fifo", seed=17,
                output=root / "out", port=8012, nvtx=False,
                python_executable="/python")
            profile_cmd = experiment.client_command(
                self.config, inputs, policy="fifo", seed=17,
                output=root / "profile", port=8012, nvtx=True,
                python_executable="/python")

            self.assertNotIn("--nvtx", cmd)
            self.assertEqual(profile_cmd.count("--nvtx"), 1)
            index = cmd.index("--nvml-device")
            self.assertEqual(cmd[index + 1], str(self.config["hardware"]["gpu_index"]))

            # A formal plan is local command construction only and never probes
            # or launches GPU processes.
            with patch.object(experiment, "inspect_gpu", side_effect=AssertionError), \
                    patch.object(experiment, "_spawn", side_effect=AssertionError), \
                    patch.object(experiment.subprocess, "Popen", side_effect=AssertionError), \
                    patch.object(experiment.subprocess, "run", side_effect=AssertionError):
                plan = experiment.render_plan(
                    self.config, inputs, port=8012, vllm_executable="/vllm",
                    python_executable="/python")
            self.assertFalse(plan["launch_performed"])
            self.assertEqual(plan["hardware_preflight"], "deferred_until_run")
            formal_stages = [stage for stage in plan["stages"]
                             if stage["phase"] in {"qualification", "opportunity", "comparison"}]
            self.assertTrue(formal_stages)
            self.assertTrue(all("--nvtx" not in stage["client_command"]
                                for stage in formal_stages))
            with self.assertRaises(experiment.ExperimentError):
                experiment.render_plan(self.config, inputs, port=8123)

    def test_client_preserves_the_active_interpreter_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = str(root / "environment" / "bin" / "python")
            inputs = _test_inputs(root)
            with patch.object(experiment.sys, "executable", interpreter), \
                    patch.object(Path, "resolve", side_effect=AssertionError("Do not dereference Python symlinks")):
                command = experiment.client_command(self.config, inputs, policy="fifo", seed=1,
                                                    output=root / "out", port=8000)
            self.assertEqual(command[0], interpreter)

    def test_profile_phase_is_the_only_runner_route_that_enables_nvtx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profile"
            output.mkdir()
            args = Namespace(
                qualification_receipt=root / "qualification.json",
                opportunity_receipt=None,
                output=root / "unused", profile_policy="asyncroll", trace_output=None)
            inputs = _test_inputs(root)
            preflight = {
                "provenance": {"config_sha256": "config-hash", "source_sha256": {}},
                "qualification_receipt_sha256": "qualification-hash",
            }
            arm = {"summary_file": "summary.json", "summary": {"completed": 32}}
            with patch.object(experiment, "validate_prerequisites", return_value=preflight), \
                    patch.object(experiment, "_new_run_directory", return_value=output), \
                    patch.object(experiment, "_write_receipt"), \
                    patch.object(experiment, "software_versions", return_value={"vllm": "test"}), \
                    patch.object(experiment, "inspect_gpu", return_value={
                        "uuid": "GPU-test", "physical_index": self.config["hardware"]["gpu_index"]}), \
                    patch.object(experiment, "validate_prerequisite_hardware"), \
                    patch.object(experiment, "run_arm", return_value=arm) as run_arm:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(experiment.run_profile(args, self.config, inputs), 0)
            self.assertIs(run_arm.call_args.kwargs["nvtx"], True)
            self.assertEqual(run_arm.call_args.kwargs["policy"], "asyncroll")

    def test_startup_audit_requires_observed_backend_and_completed_graph_capture(self):
        actual_evidence = experiment.audit_startup_log(
            "INFO Using FlashAttention backend\n"
            "INFO CUDAGraph capture finished in 2.1 seconds\n")
        self.assertEqual(actual_evidence["status"], "verified")
        self.assertTrue(actual_evidence["attention_backend_verified"])
        self.assertTrue(actual_evidence["cuda_graph_capture_verified"])

        requested_only = experiment.audit_startup_log(
            "argv: --attention-backend FLASH_ATTN --enforce-eager=False\n")
        self.assertEqual(requested_only["status"], "unverified")
        self.assertFalse(requested_only["attention_backend_verified"])
        self.assertFalse(requested_only["cuda_graph_capture_verified"])

        explicit_negative = experiment.audit_startup_log(
            "INFO Using FlashAttention backend\n"
            "INFO CUDA graphs are disabled\n"
            "INFO CUDAGraph capture finished for a warmup shape\n")
        self.assertFalse(explicit_negative["cuda_graph_capture_verified"])

        unrelated_backend = experiment.audit_startup_log(
            "INFO Using XGRAMMAR backend\n"
            "INFO CUDAGraph capture finished for a warmup shape\n")
        self.assertEqual(unrelated_backend["status"], "unverified")
        self.assertFalse(unrelated_backend["attention_backend_verified"])

    def test_qualification_requires_actual_tool_use_and_both_telemetry_sources(self):
        summary = {
            "count": 2, "completed": 2, "failed": 0,
            "action_valid_rate": 1.0, "trajectory_tool_call_rate": 1.0,
            "results": [
                {"status": "completed", "tool_calls": 1},
                {"status": "completed", "tool_calls": 2},
            ],
            "telemetry_probe": {"availability": {"vllm": True, "nvml": True}},
        }
        passed, checks = experiment.qualification_pass(summary, 2)
        self.assertTrue(passed)
        self.assertTrue(checks["every_task_has_tool_call"])
        self.assertTrue(checks["vllm_metrics_available"])
        self.assertTrue(checks["nvml_available"])

        missing_nvml = copy.deepcopy(summary)
        missing_nvml["telemetry_probe"]["availability"]["nvml"] = False
        self.assertFalse(experiment.qualification_pass(missing_nvml, 2)[0])
        no_tools = copy.deepcopy(summary)
        no_tools["results"][0]["tool_calls"] = 0
        no_tools["trajectory_tool_call_rate"] = 0.5
        self.assertFalse(experiment.qualification_pass(no_tools, 2)[0])
        self.assertFalse(experiment.qualification_pass({"count": 2}, 2)[0])

    def test_opportunity_gate_rejects_missing_telemetry_and_nominal_tiny_queue(self):
        gate = self.config["opportunity_gate"]
        summary = {
            "count": 8, "completed": 8, "failed": 0,
            "action_valid_rate": 1.0, "trajectory_tool_call_rate": 1.0,
            "results": [{"status": "completed", "tool_calls": 1} for _ in range(8)],
            "telemetry_probe": {"availability": {"vllm": True, "nvml": True}},
            "elapsed_seconds": 100,
            "phase_seconds": {"tool_queue": {"p95_seconds": 0.001}},
            "resource_metrics": {
                "cpu_related_idle_candidate_seconds": 0.0005,
                "client_cpu_blocked_seconds": 0.0005,
                "cpu_max_queue_depth": 1,
                "joint_observation_coverage_seconds": 1,
            },
        }
        passed, checks = experiment.opportunity_pass(summary, gate)
        self.assertFalse(passed)
        self.assertFalse(checks["cpu_max_queue_depth"]["pass"])
        self.assertFalse(checks["tool_queue_p95_seconds"]["pass"])
        self.assertFalse(experiment.opportunity_pass({"count": 8}, gate)[0])

    def test_arbitrary_model_folder_is_accepted_only_after_full_pinned_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "arbitrary-folder-name"
            model_path.mkdir()
            contents = {
                "config.json": b'{"hidden_size": 2}',
                "model.safetensors.index.json": (
                    b'{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}'),
                "model-00001-of-00001.safetensors": b"pinned weights",
            }
            for name, data in contents.items():
                (model_path / name).write_bytes(data)
            manifest = {
                "repository": self.config["model"]["repository"],
                "revision": self.config["model"]["revision"],
                "files": [{
                    "name": name, "size": len(data), "algorithm": "sha256",
                    "digest": hashlib.sha256(data).hexdigest(),
                } for name, data in contents.items()],
            }
            with patch.object(experiment, "_load_model_manifest", return_value=manifest):
                pending = experiment.verify_model_snapshot(
                    model_path, self.config, full_hashes=False)
                verified = experiment.verify_model_snapshot(
                    model_path, self.config, full_hashes=True)
            self.assertTrue(pending["complete"])
            self.assertFalse(pending["weights_revision_verified"])
            self.assertTrue(verified["weights_revision_verified"])
            self.assertEqual(verified["hash_verification_status"], "all_content_digests_verified")

            # Keep the file size and arbitrary directory name unchanged while
            # proving that content, rather than the path label, is authoritative.
            (model_path / "model-00001-of-00001.safetensors").write_bytes(b"tamper weights")
            with patch.object(experiment, "_load_model_manifest", return_value=manifest):
                tampered = experiment.verify_model_snapshot(
                    model_path, self.config, full_hashes=True)
            self.assertFalse(tampered["complete"])
            self.assertFalse(tampered["weights_revision_verified"])

    def test_failed_arm_keeps_both_pids_and_owned_group_cleanup_receipt(self):
        class FakeProcess:
            def __init__(self, pid, exit_code=None):
                self.pid = pid
                self.exit_code = exit_code
                self.returncode = None

            def poll(self):
                return None

            def wait(self, timeout=None):
                self.returncode = self.exit_code
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _test_inputs(root)
            gpu = {"uuid": "GPU-physical-test", "driver_version": "test-driver",
                   "physical_index": self.config["hardware"]["gpu_index"]}
            server = FakeProcess(41001)
            client = FakeProcess(41002, exit_code=7)
            spawned = []

            def fake_spawn(command, log_path, env, cwd):
                spawned.append((command, dict(env)))
                return server if len(spawned) == 1 else client

            with patch.object(experiment, "_port_available", return_value=True), \
                    patch.object(experiment, "inspect_gpu", side_effect=[gpu, gpu]), \
                    patch.object(experiment, "_http_ready", return_value=True), \
                    patch.object(experiment, "audit_startup_log", return_value={
                        "status": "verified", "attention_backend_verified": True,
                        "cuda_graph_capture_verified": True}), \
                    patch.object(experiment, "server_command", return_value=["/vllm", "serve"]), \
                    patch.object(experiment, "_spawn", side_effect=fake_spawn), \
                    patch.object(experiment, "terminate_owned_group") as terminate, \
                    patch.object(experiment, "_close_process_stream"):
                with self.assertRaises(experiment.ExperimentError):
                    experiment.run_arm(
                        self.config, inputs, output=root / "arm", policy="fifo",
                        seed=5, limit=8, gpu=gpu)

            arm_receipt = json.loads((root / "arm" / "arm.json").read_text(encoding="utf-8"))
            self.assertEqual(arm_receipt["status"], "failed")
            self.assertEqual((arm_receipt["server_pid"], arm_receipt["client_pid"]),
                             (server.pid, client.pid))
            self.assertTrue(arm_receipt.get("cleanup_completed_utc"))
            self.assertEqual(terminate.call_count, 2)
            self.assertEqual(
                spawned[0][1]["CUDA_VISIBLE_DEVICES"], str(gpu["physical_index"])
            )
            self.assertEqual(arm_receipt["gpu"]["uuid"], gpu["uuid"])
            self.assertEqual(
                arm_receipt["cuda_visible_devices"], str(gpu["physical_index"])
            )
            client_command = spawned[1][0]
            self.assertNotIn("--nvtx", client_command)
            nvml_arg = client_command.index("--nvml-device")
            self.assertEqual(client_command[nvml_arg + 1], str(gpu["physical_index"]))


if __name__ == "__main__":
    unittest.main()
