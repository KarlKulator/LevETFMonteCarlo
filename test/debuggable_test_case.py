import sys
import unittest

class DebuggableTestCase(unittest.TestCase):
    def __add_error_replacement(self, _, err):
        value, traceback = err[1:]
        raise value.with_traceback(traceback)

    def run(self, result=None):
        if result and sys.gettrace() is not None:
            result.addError = self.__add_error_replacement
        super().run(result)

