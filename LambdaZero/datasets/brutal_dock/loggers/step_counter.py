class StepCounter:
    """
    This class keeps track of how many times a run_id has been called.
    """

    def __init__(self):
        self._count = 0

    @property
    def count(self):
        return self._count

    def increment_and_return_count(self):
        self._count += 1
        return self._count