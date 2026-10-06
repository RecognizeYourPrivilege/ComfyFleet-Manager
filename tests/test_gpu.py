import subprocess
import unittest

from comfyfleet.errors import FleetError
from comfyfleet.gpu import Gpu, detect_gpus, parse_nvidia_smi, select_gpus


class GpuTests(unittest.TestCase):
    def test_parse_csv(self):
        gpus = parse_nvidia_smi('0, NVIDIA GeForce RTX 4090, 24564 MiB\n1, "Name, With Comma", 1024 MiB\n')
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].name, "NVIDIA GeForce RTX 4090")
        self.assertEqual(gpus[1].name, "Name, With Comma")

    def test_missing_nvidia_smi(self):
        def run(_argv):
            raise FileNotFoundError("nvidia-smi")

        with self.assertRaises(FleetError) as ctx:
            detect_gpus(run=run)
        self.assertIn("nvidia-smi", str(ctx.exception))

    def test_failing_nvidia_smi(self):
        def run(argv):
            return subprocess.CompletedProcess(argv, 1, "", "NVIDIA-SMI has failed")

        with self.assertRaises(FleetError) as ctx:
            detect_gpus(run=run)
        self.assertIn("failed", str(ctx.exception).lower())

    def test_multi_gpu_noninteractive_requires_a_flag(self):
        gpus = [Gpu(0, "A", "1 MiB"), Gpu(1, "B", "1 MiB")]
        with self.assertRaises(FleetError) as ctx:
            select_gpus(gpus, interactive=False)
        self.assertIn("GPU", str(ctx.exception))

    def test_multi_gpu_prompt_rejects_empty(self):
        gpus = [Gpu(0, "A", "1 MiB"), Gpu(1, "B", "1 MiB")]
        with self.assertRaises(FleetError):
            select_gpus(gpus, interactive=True, prompt=lambda _message: "  ")

    def test_multi_gpu_prompt_accepts_several(self):
        gpus = [Gpu(0, "A", "1 MiB"), Gpu(1, "B", "1 MiB")]
        chosen = select_gpus(gpus, interactive=True, prompt=lambda _message: "0,1")
        self.assertEqual(chosen, [0, 1])

    def test_flag_all(self):
        gpus = [Gpu(0, "A", ""), Gpu(2, "C", "")]
        self.assertEqual(select_gpus(gpus, gpus_spec="all", interactive=False), [0, 2])

    def test_single_gpu_must_be_confirmed(self):
        gpus = [Gpu(0, "Only", "8 MiB")]
        with self.assertRaises(FleetError):
            select_gpus(gpus, interactive=True, prompt=lambda _message: "n")
        self.assertEqual(
            select_gpus(gpus, interactive=True, prompt=lambda _message: "y"),
            [0],
        )


if __name__ == "__main__":
    unittest.main()
