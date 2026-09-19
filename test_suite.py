import sys
import unittest

# Ensure UTF-8 output encoding for Windows terminal / PowerShell compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import target functions from app.py
from app import calculate, ALLOWED_TOOLS, TOOLS


class TestCalculatorTool(unittest.TestCase):

    """Unit test cases for calculate function and tool definitions."""

    def test_addition(self):
        res = calculate("add", 15, 27)
        self.assertEqual(res, {"result": 42})

    def test_subtraction(self):
        res = calculate("subtract", 100, 45)
        self.assertEqual(res, {"result": 55})

    def test_multiplication(self):
        res = calculate("multiply", 12, 8)
        self.assertEqual(res, {"result": 96})

    def test_division(self):
        res = calculate("divide", 144, 12)
        self.assertEqual(res, {"result": 12})

    def test_division_by_zero(self):
        res = calculate("divide", 50, 0)
        self.assertIn("error", res)
        self.assertEqual(res["error"], "Division by zero is not allowed.")

    def test_invalid_operation(self):
        res = calculate("power", 2, 3)
        self.assertIn("error", res)
        self.assertTrue("Invalid operation" in res["error"])

    def test_invalid_arguments(self):
        res = calculate("add", "invalid_num", 10)
        self.assertIn("error", res)

    def test_tool_schema_structure(self):
        self.assertEqual(len(TOOLS), 1)
        tool = TOOLS[0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "calculate")
        params = tool["function"]["parameters"]
        self.assertIn("operation", params["properties"])
        self.assertIn("a", params["properties"])
        self.assertIn("b", params["properties"])
        self.assertEqual(params["required"], ["operation", "a", "b"])

    def test_allowed_tools_whitelist(self):
        self.assertIn("calculate", ALLOWED_TOOLS)
        self.assertEqual(len(ALLOWED_TOOLS), 1)
        self.assertEqual(ALLOWED_TOOLS["calculate"], calculate)


if __name__ == "__main__":
    print("=====================================================")
    print("🧪 Running Unit Tests for AI Calculator Agent...")
    print("=====================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculatorTool)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
