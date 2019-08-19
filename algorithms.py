import abc
import random
import numpy as np
import pandas as ps
from exceptions import DtypeNotSupported
from scipy.stats import entropy


from decision_tree import DecisionTree, Node


class TDIDT(abc.ABC):

    def _choose_attribute(self, E, T):
        raise NotImplementedError()

    def _split_by_attribute(self, E: ps.DataFrame, T: ps.Series, attribute: str)->(ps.DataFrame, ps.DataFrame, str):
        raise NotImplementedError()

    def fit(self, E, T):
        decision_tree = DecisionTree()
        self._fit_aux(decision_tree.root, E, T)
        return decision_tree

    def _fit_aux(self, node, E, T):
        if E.empty():
            return node
        # check if all examples are of the same class - i.e. it is a leaf
        for t in T:
            if t != T[0]:
                return Node(None, None, node, None, None)
        attribute, threshold = self._choose_attribute(E, T)  # should return the threshold
        node.attribute = attribute
        E_left, E_right = self._split_by_attribute(E, T, attribute, threshold)
        node.threshold = threshold
        T_new = T.copy.remove(attribute)  # TODO: review
        left_son = Node(None, None, node, None, None)
        right_son = Node(None, None, node, None, None)
        self._fit_aux(left_son, E_left, T_new)
        self._fit_aux(right_son, E_right, T_new)


class RandomChooseTree(TDIDT):

    def _choose_attribute(self, E, T):
        return random.choice(T)  # todo: review


class SplitterExampleTree(TDIDT):

    def _split_by_attribute(self, E: ps.DataFrame, T: ps.Series, attribute: str)->(ps.DataFrame, ps.DataFrame, str):
        att_dtype = E[attribute].dtype
        if att_dtype in E.select_dtypes(include=np.number):
            split_function = self._split_by_numerical
        elif att_dtype in E.select_dtypes(include=object):
            split_function = self._split_by_categorical
        else:
            raise DtypeNotSupported(att_dtype)
        return split_function(E, attribute)

    def _split_by_numerical(self, data: ps.DataFrame, attribute: str) -> (
        ps.DataFrame, ps.DataFrame, float):
        feature_col = data[attribute]
        mean = feature_col.mean()
        left_son = data.loc[feature_col < mean]
        right_son = data.loc[feature_col >=mean]
        threshold = mean
        return left_son, right_son, str(threshold)

    def _split_by_categorical(self, data: ps.DataFrame, attribute: str) -> (
        ps.DataFrame, ps.DataFrame, float):
        feature_col = data[attribute]
        chosen_category = feature_col.loc[random.randint(0, len(data))]
        left_son = data[feature_col == chosen_category]
        right_son = data[feature_col == chosen_category]
        threshold = chosen_category
        return left_son, right_son, threshold
