"""
This is just a trivial test file to get the testing plumbing in place. It will be removed once everything works.
"""
from LambdaZero.datasets.brutal_dock.trivial import one_plus_one


def test_one_plus_one():
    assert one_plus_one() == 2
