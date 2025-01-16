class Behavior:
    def __init__(self, name, onset_times, offset_times, data_type='epocs'):
        """
        Initialize a Behavior object.

        Args:
            name (str): Name of the behavior.
            onset_times (list): List of onset times (in seconds).
            offset_times (list): List of offset times (in seconds).
            data_type (str): Type of the behavior, default is 'epocs'.
        """
        self.name = name
        self.onset_times = onset_times
        self.offset_times = offset_times
        self.data_type = data_type
        self.data = [1] * len(onset_times)