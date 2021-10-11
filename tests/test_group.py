import unittest
from cal.group import run_samples, _logging


class TestGroup(unittest.TestCase):

    def test_group_from_table(self):
        run_samples(_logging.ERROR)
