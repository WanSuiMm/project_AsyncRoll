import tempfile
import unittest
from pathlib import Path

from asyncroll.tools import call_tool


class ToolTest(unittest.TestCase):
    def test_integer_dictionary_keys_are_restored_after_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "product.py"
            path.write_text(
                "def product(values):\n"
                "    return sum(key * value for key, value in values.items())\n",
                encoding="utf-8",
            )
            tool = {
                "name": "product", "implementation": str(path),
                "inputs": {"values": "dict[int, int]"},
            }
            self.assertEqual(call_tool(tool, {"values": {"2": 3, "5": 4}}), 26)

    def test_declared_inputs_are_enforced(self):
        tool = {"name": "unused", "inputs": {"value": "int"},
                "implementation": "missing.py"}
        with self.assertRaises(TypeError):
            call_tool(tool, {"other": 1})


if __name__ == "__main__":
    unittest.main()
