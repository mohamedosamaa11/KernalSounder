import unittest

from nlp_engine import parse_command
from security import Intent


class ParseCommandTests(unittest.TestCase):
    def check(self, text, intent, **slots):
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, intent, f"{text!r} -> {parsed}")
        for key, value in slots.items():
            self.assertEqual(parsed.slots.get(key), value, f"{text!r} -> {parsed.slots}")
        return parsed

    def test_whisper_style_output(self):
        self.check("create a file called notes.txt.", Intent.CREATE_FILE, name="notes.txt")
        self.check("rename old.txt to new.txt.", Intent.RENAME_FILE, source="old.txt", destination="new.txt")
        self.check("delete file report.txt.", Intent.DELETE_FILE, name="report.txt")
        self.check("show me the cpu, ram and disk usage.", Intent.SHOW_USAGE)
        self.check("close firefox please.", Intent.CLOSE_APP, app="firefox")

    def test_spoken_dot(self):
        self.check("create file called notes dot txt", Intent.CREATE_FILE, name="notes.txt")

    def test_wake_word_is_stripped(self):
        self.check("hey Jarvis, open the browser.", Intent.OPEN_APP, app="browser")
        self.check("Jarvis list files", Intent.LIST_FILES)
        self.check("High sounder. Open the browser.", Intent.OPEN_APP, app="browser")
        self.check("Hi Sounder, list files", Intent.LIST_FILES)
        self.check("Hey Nova, open the browser.", Intent.OPEN_APP, app="browser")
        self.check("Hey, Nova. List files.", Intent.LIST_FILES)

    def test_fuzzy_repair(self):
        self.check("crate a file called demo.txt", Intent.CREATE_FILE, name="demo.txt")
        self.check("open fierfox", Intent.OPEN_APP, app="firefox")
        self.check("dellete file report.txt", Intent.DELETE_FILE, name="report.txt")

    def test_names_are_not_fuzzed(self):
        self.check("create folder called projects", Intent.CREATE_FOLDER, name="projects")
        self.check("create folder called labs", Intent.CREATE_FOLDER, name="labs")

    def test_sentence_period_splits(self):
        self.check("Open the browser. Thanks.", Intent.OPEN_APP, app="browser")

    def test_confirm_cancel(self):
        self.check("yes", Intent.CONFIRM)
        self.check("Yes, do it.", Intent.CONFIRM)
        self.check("confirm", Intent.CONFIRM)
        self.check("no, cancel", Intent.CANCEL)

    def test_windows_apps(self):
        self.check("open notepad", Intent.OPEN_APP, app="notepad")
        self.check("open calculator", Intent.OPEN_APP, app="calculator")

    def test_bare_words(self):
        self.check("videos", Intent.OPEN_APP)
        self.check("what apps are available", Intent.LIST_APPS)
        self.check("show processes", Intent.SHOW_PROCESSES)

    def test_dangerous_text_is_refused(self):
        for text in ("rm -rf /", "sudo shutdown now", "format c:"):
            self.assertEqual(parse_command(text).intent, Intent.DANGEROUS, text)


if __name__ == "__main__":
    unittest.main()
