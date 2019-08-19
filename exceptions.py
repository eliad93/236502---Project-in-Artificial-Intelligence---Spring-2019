import numpy as np
import pandas as ps

class Error(Exception):
    pass


class DtypeNotSupported(Error):

    def __init__(self, dtype):
        self.data_type = dtype
        print(f"{dtype} is not supported ")
