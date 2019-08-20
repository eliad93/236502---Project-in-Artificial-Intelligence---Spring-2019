import abc
import random
import numpy as np
import pandas as ps
from exceptions import DtypeNotSupported
import copy
from scipy.stats import entropy

from decision_tree import DecisionTree, Node


class TDIDT(abc.ABC):

    def __init__(self):
        self.decision_tree = None

    def _choose_attribute(self, E, T):
        raise NotImplementedError()

    def _split_by_attribute(self, E: ps.DataFrame, T: ps.Series,
                            attribute: str) -> (
    ps.DataFrame, ps.DataFrame, str):
        raise NotImplementedError()

    def fit(self, E, T):
        decision_tree = DecisionTree()
        T_new = copy.deepcopy(T)
        E_new = copy.deepcopy(E)
        self._fit_aux(decision_tree.root, E_new, T_new)
        self.decision_tree = decision_tree

    def _fit_aux(self, node, E, T):
        if E.empty or T.empty:
            return node
        # check if all examples are of the same class - i.e. it is a leaf
        tag_sample = T.sample(1).item()
        if len(T[ tag_sample== T]) == len(T):
            node.tag = tag_sample
            return node
        attribute = self._choose_attribute(E, T)  # should return the threshold
        node.attribute = attribute
        E_left, T_left, E_right, T_right, threshold = self._split_by_attribute(
            E, T, attribute)
        node.threshold = threshold
        E_left.drop(columns=attribute)
        E_right.drop(columns=attribute)
        node.left_son = Node()
        node.right_son = Node()
        self._fit_aux(node.left_son, E_left, T_left)
        self._fit_aux(node.right_son, E_right, T_right)


class RandomChooseTree(TDIDT):

    def _choose_attribute(self, E, T):
        return random.choice(T)  # todo: review


class SplitterExampleTree(TDIDT):

    def _split_by_attribute(self, E: ps.DataFrame, T: ps.Series,
                            attribute: str) -> (
    ps.DataFrame, ps.Series, ps.DataFrame, ps.Series, str):
        if attribute in E.select_dtypes(include=np.number):
            split_function = self._split_by_numerical
        elif attribute in E.select_dtypes(include=object):
            split_function = self._split_by_categorical
        else:
            raise DtypeNotSupported()
        return split_function(E, T, attribute)

    def _split_by_numerical(self, data: ps.DataFrame, tags: ps.Series,
                            attribute: str) -> (
    ps.DataFrame, ps.Series, ps.DataFrame, ps.Series, str):
        feature_col = data[attribute]
        mean = feature_col.mean()
        left_indices = feature_col < mean
        right_indices = feature_col >= mean
        left_son = data.loc[left_indices]
        left_tags = tags.loc[left_indices]
        right_son = data.loc[right_indices]
        right_tags = tags.loc[right_indices]
        threshold = mean
        return left_son, left_tags, right_son, right_tags, str(threshold)

    def _split_by_categorical(self, data: ps.DataFrame, tags: ps.Series,
                              attribute: str) -> (
    ps.DataFrame, ps.Series, ps.DataFrame, ps.Series, str):
        feature_col = data[attribute]
        chosen_category = feature_col.loc[random.randint(0, len(data))]
        left_indices = feature_col == chosen_category
        right_indices = feature_col != chosen_category
        left_son = data[left_indices]
        left_tags = tags[left_indices]
        right_son = data[right_indices]
        right_tags = tags[right_indices]
        threshold = chosen_category
        return left_son, left_tags, right_son, right_tags, threshold

    def _choose_attribute(self, E: ps.DataFrame, T: ps.Series):
        feature_name_list = list(E.columns)
        target_name = T.name
        feature_name_list.remove(target_name)
        return random.choice(feature_name_list)
