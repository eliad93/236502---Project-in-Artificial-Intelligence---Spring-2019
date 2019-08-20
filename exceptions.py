import numpy as np
import pandas as ps

class Error(Exception):
    pass


class DtypeNotSupported(Error):

    def __init__(self, ):
        print(f"data type is not supported ")
