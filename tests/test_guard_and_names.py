import unittest

from nlp_engine import extract_name_answer, parse_command
from security import Intent, ValidationState, validate

REFUSE = [
    "delete system", "delete the system files", "remove windows", "delete everything",
    "delete all files", "format the c drive", "shut down the computer", "restart the pc",
    "sudo delete file x", "run as administrator", "open registry editor", "uninstall spotify",
    "install telegram", "rm -rf /", "run this command", "delete my documents",
    "empty the recycle bin", "disable the firewall", "close explorer", "kill all",
    "what is my password", "wipe the hard disk", "delete the workspace",
]
ALLOW = [
    "delete file report.txt", "remove the folder labs", "create file named Mohamed",
    "close firefox", "open spotify", "show cpu ram and disk usage", "show processes",
    "list files", "open file explorer", "rename old.txt to new.txt", "open terminal",
    "close the terminal", "delete file system_notes.txt", "show all processes",
]


class GuardTests(unittest.TestCase):
    def test_refusals(self):
        for text in REFUSE:
            parsed = parse_command(text)
            self.assertEqual(parsed.intent, Intent.DANGEROUS, text)
            result = validate(parsed)
            self.assertEqual(result.state, ValidationState.BLOCKED, text)
            self.assertTrue(result.reason, text)

    def test_no_false_refusals(self):
        for text in ALLOW:
            self.assertNotEqual(parse_command(text).intent, Intent.DANGEROUS, text)


class NameTests(unittest.TestCase):
    def check(self, text, intent, name):
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, intent, text)
        self.assertEqual(parsed.slots.get("name"), name, text)

    def test_name_phrasings_and_case(self):
        self.check("Create a file named Mohamed.", Intent.CREATE_FILE, "Mohamed")
        self.check("create file. its name is Anas.", Intent.CREATE_FILE, "Anas")
        self.check("create a new file and call it Report.txt", Intent.CREATE_FILE, "Report.txt")
        self.check("make a folder with the name Projects", Intent.CREATE_FOLDER, "Projects")
        self.check("create file name Sara", Intent.CREATE_FILE, "Sara")
        self.check("Create a file called Notes.TXT", Intent.CREATE_FILE, "Notes.txt")
        self.check("Create a file named M-O-H-A-M-E-D.", Intent.CREATE_FILE, "Mohamed")
        self.check("create a file named Mohammed", Intent.CREATE_FILE, "Mohamed")
        self.check("create a file named sarah.txt", Intent.CREATE_FILE, "Sara.txt")
        self.check("create a file named report", Intent.CREATE_FILE, "report")

    def test_missing_name_is_low_confidence(self):
        for text in ("create file", "create a file.", "create a file name."):
            parsed = parse_command(text)
            self.assertEqual(parsed.intent, Intent.CREATE_FILE, text)
            self.assertFalse(parsed.slots.get("name"), text)
            self.assertLess(parsed.confidence, 0.45, text)

    def test_follow_up_answers(self):
        self.assertEqual(extract_name_answer("Mohamed."), "Mohamed")
        self.assertEqual(extract_name_answer("Call it Mohamed."), "Mohamed")
        self.assertEqual(extract_name_answer("Okay, name it Sara please."), "Sara")
        self.assertEqual(extract_name_answer("M O H A M E D"), "Mohamed")
        self.assertEqual(extract_name_answer("notes dot txt"), "notes.txt")


class CatalogMatchTests(unittest.TestCase):
    def test_fuzzy_and_denied(self):
        from app_catalog import AppCatalog, AppEntry, DENY_PATTERNS

        cat = AppCatalog()
        cat._apps = [AppEntry("Spotify", ["x"]), AppEntry("Discord", ["x"]),
                     AppEntry("Visual Studio Code", ["x"]), AppEntry("Registry Editor", ["x"]),
                     AppEntry("VLC media player - reset preferences and cache files", ["x"])]
        for app in cat._apps:
            app.denied = bool(DENY_PATTERNS.search(app.name))
        cat._loaded.set()
        cat._loaded_at = 1e18
        self.assertEqual(cat.find("spot fire")[0].name, "Spotify")
        self.assertEqual(cat.find("vs code")[0].name, "Visual Studio Code")
        self.assertTrue(cat.find("registry editor")[0].denied)
        self.assertTrue(cat._apps[4].denied)
        self.assertIsNone(cat.find("telegram"))
        self.assertEqual(cat.find("spatifia")[0].name, "Spotify")


if __name__ == "__main__":
    unittest.main()
