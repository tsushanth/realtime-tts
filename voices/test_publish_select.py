"""Offline tests for publish_voice.select (no network, no Modal).  python3 voices/test_publish_select.py"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import publish_voice as pv


class Select(unittest.TestCase):
    def test_tier_a_skips_already_published(self):
        ids = pv.select(["--tier", "A"])
        self.assertTrue(ids)
        self.assertFalse(set(ids) & pv.ALREADY_PUBLISHED)
        cat = {v["id"]: v for v in pv.load_catalog()}
        self.assertTrue(all(cat[i]["tier"] == "A" for i in ids))

    def test_tier_b_needs_flag(self):
        with self.assertRaises(SystemExit):
            pv.select(["--tier", "B"])
        ids = pv.select(["--tier", "B", "--lang", "es", "--allow-tier-b"])
        self.assertTrue(ids and all(i.startswith("es-") for i in ids))

    def test_tier_c_never(self):
        with self.assertRaises(SystemExit):
            pv.select(["--tier", "C"])

    def test_republish_flag(self):
        self.assertIn("de-de-mls", pv.select(["de-de-mls", "--republish"]))
        self.assertNotIn("de-de-mls", pv.select(["de-de-mls"]))

    def test_lang_filter(self):
        ids = pv.select(["--tier", "A", "--lang", "fr"])
        self.assertTrue(ids and all(i.startswith("fr-") for i in ids))


if __name__ == "__main__":
    unittest.main()
