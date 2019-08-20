
class Node:

    def __init__(self, left_son=None, right_son=None, attribute="", threshold=None):
        self.left_son = left_son
        self.right_son = right_son
        self.attribute = attribute
        self.threshold = threshold
        self.tag = None

    def is_leaf(self):
        return self.left_son is None and self.right_son is None and \
               self.attribute is None and self.threshold is None and \
               self.tag is not None


class DecisionTree:

    def __init__(self):
        self.root = Node()
        self.size = 0

    def predict(self):
        pass
