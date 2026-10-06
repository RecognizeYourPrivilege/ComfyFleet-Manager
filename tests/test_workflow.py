import json
import tempfile
import unittest
from pathlib import Path

from comfyfleet.errors import FleetError
from comfyfleet.workflow import load_operator_workflow


class WorkflowTests(unittest.TestCase):
    def test_requires_json_object_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.json"
            path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
            data = load_operator_workflow(path)
            self.assertEqual(data["nodes"], [])

    def test_missing_file(self):
        with self.assertRaises(FleetError):
            load_operator_workflow(Path("/no/such/comfyfleet-flow.json"))

    def test_rejects_non_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(FleetError):
                load_operator_workflow(path)

    def test_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.json"
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(FleetError):
                load_operator_workflow(path)

    def test_rejects_non_json_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.txt"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(FleetError) as ctx:
                load_operator_workflow(path)
            self.assertIn("no stock default", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
