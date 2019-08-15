import abc
import random

from decision_tree import DecisionTree, Node


class TDIDT(abc.ABC):

    def _choose_attribute(self, E, T):
        raise NotImplementedError()

    def _split_by_attribute(self, E, T, attribute):
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
        attribute = self._choose_attribute(E, T)
        node.attribute = attribute
        E_left, E_right, threshold = self._split_by_attribute(E, T, attribute)
        node.threshold = threshold
        T_new = T.copy.remove(attribute)
        left_son = Node(None, None, node, None, None)
        right_son = Node(None, None, node, None, None)
        self._fit_aux(left_son, E_left, T_new)
        self._fit_aux(right_son, E_right, T_new)


class RandomChooseTree(TDIDT):

    def _choose_attribute(self, E, T):
        return random.choice(T)
