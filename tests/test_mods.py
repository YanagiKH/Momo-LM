import tempfile
import textwrap
import unittest
from pathlib import Path

from momo_lm.mods import ModManager


class ModTests(unittest.TestCase):
    def test_mod_command_loads_without_breaking_other_mods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "working.py").write_text(
                textwrap.dedent(
                    """
                    from momo_lm.mods import ModSpec
                    def register():
                        return ModSpec(name="Test", commands={"/ping": lambda argument: "pong " + argument})
                    """
                ),
                encoding="utf-8",
            )
            (root / "broken.py").write_text("raise RuntimeError('broken by design')", encoding="utf-8")
            manager = ModManager(root)
            with self.assertLogs("momo_lm.mods", level="ERROR"):
                manager.load()
            self.assertEqual(manager.command("/ping ok"), "pong ok")
            self.assertEqual(len(manager.errors), 1)


if __name__ == "__main__":
    unittest.main()
