from unittest import TestCase
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from algorithms import SplitterExampleTree

def load_iris_as_df(load_function)->pd. DataFrame:
    iris = load_function()
    data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
    return data1

class TestSplitterExampleTree(TestCase):


    def test__split_by_attribute(self):
        self.fail()

    def test__split_by_numerical(self):
        self.fail()

    def test__split_by_categorical(self):
        self.fail()

    def test__choose_attribute(self):
        self.fail()

    def test_SplitterExampleTree(self):
        iris = load_iris_as_df(load_iris)
        tree = SplitterExampleTree()
        tree.fit(iris, iris['target'])
        x = iris
