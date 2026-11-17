import unittest

from config import WORKSPACE
from executor import execute
from nlp_engine import parse_command
from security import Intent, ValidationState, ensure_workspace, validate


def run(text):
    parsed = parse_command(text)
    result = validate(parsed)
    return parsed, result


class SecurityTests(unittest.TestCase):
    def test_traversal_and_blocked_names(self):
        for text in ("create file called ../../etc/passwd", "create file called id_rsa",
                     "create file called con.txt", "create folder called sudoers"):
            _, result = run(text)
            self.assertEqual(result.state, ValidationState.BLOCKED, text)

    def test_delete_needs_confirmation(self):
        ensure_workspace()
        (WORKSPACE / "victim.txt").write_text("x")
        _, result = run("delete file victim.txt")
        self.assertEqual(result.state, ValidationState.NEEDS_CONFIRM)

    def test_unknown_app_blocked(self):
        _, result = run("open regedit")
        self.assertEqual(result.state, ValidationState.BLOCKED)

    def test_system_shell_cannot_be_closed_on_windows(self):
        from config import IS_WINDOWS
        if not IS_WINDOWS:
            self.skipTest("Windows only")
        _, result = run("close file explorer")
        self.assertEqual(result.state, ValidationState.BLOCKED)


class ExecutorTests(unittest.TestCase):
    def test_file_lifecycle(self):
        ensure_workspace()
        for leftover in ("alpha.txt", "beta.txt"):
            (WORKSPACE / leftover).unlink(missing_ok=True)

        parsed, result = run("create a file called alpha.txt")
        self.assertTrue(execute(parsed, result).ok)
        self.assertTrue((WORKSPACE / "alpha.txt").is_file())

        parsed, result = run("create a file called alpha.txt")
        self.assertEqual(result.state, ValidationState.BLOCKED)

        parsed, result = run("rename alpha.txt to beta.txt")
        self.assertTrue(execute(parsed, result).ok)
        self.assertTrue((WORKSPACE / "beta.txt").is_file())

        parsed, result = run("list files")
        outcome = execute(parsed, result)
        self.assertIn("beta.txt", outcome.details)

        parsed, result = run("delete file beta.txt")
        self.assertEqual(result.state, ValidationState.NEEDS_CONFIRM)
        result.state = ValidationState.SAFE
        self.assertTrue(execute(parsed, result).ok)
        self.assertFalse((WORKSPACE / "beta.txt").exists())

    def test_folder_must_be_empty(self):
        ensure_workspace()
        folder = WORKSPACE / "labs"
        folder.mkdir(exist_ok=True)
        (folder / "keep.txt").write_text("x")
        parsed, result = run("delete folder labs")
        result.state = ValidationState.SAFE
        outcome = execute(parsed, result)
        self.assertFalse(outcome.ok)
        self.assertTrue(folder.exists())

    def test_read_only_snapshots(self):
        for text, intent in (("show cpu ram and disk usage", Intent.SHOW_USAGE),
                             ("show processes", Intent.SHOW_PROCESSES)):
            parsed, result = run(text)
            self.assertEqual(parsed.intent, intent)
            outcome = execute(parsed, result)
            self.assertTrue(outcome.ok, outcome.details)


if __name__ == "__main__":
    unittest.main()
