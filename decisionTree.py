import pandas as pd
import math
from typing import Dict, TypeVar, List, Any, NamedTuple, Optional, Union
from collections import defaultdict, Counter

df = pd.read_csv('recrutation.csv', usecols=['Level', 'Programming Language', 'Social_media_active', 'Higher education',
                                             'Good interview impression'])
data_to_list = df.values.tolist()
inputs = []


class Candidate(NamedTuple):
    level: str
    lang: str
    socialMedia: bool
    education: bool
    did_well: Optional[bool] = None


for data in data_to_list:
    a, b, c, d, e = data
    inputs.append(Candidate(a, b, c, d, e))

T = TypeVar('T')

'''Entropy'''


def entropy(class_probalities: List[float]) -> float:
    return sum(-p * math.log(p, 2) for p in class_probalities if p > 0)


def class_probalities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probalities(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)


'''Division function'''


def partition_by(inputs: List[T], attr: str) -> Dict[Any, List[T]]:
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for i in inputs:
        key = getattr(i, attr)
        partitions[key].append(i)
    return partitions


'''Calculate entropy function'''


def partition_entropy_by(inputs: List[Any], attr: str, label_attr: str) -> float:
    partitions = partition_by(inputs, attr)
    labels = [[getattr(i, label_attr) for i in partition] for partition in partitions.values()]
    return partition_entropy(labels)


class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attr: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]


def classify(tree: DecisionTree, i: Any) -> Any:
    if isinstance(tree, Leaf):
        return tree.value
    subtree_key = getattr(i, tree.attr)

    if subtree_key not in tree.subtrees:
        return tree.default_value
    subtree = tree.subtrees[subtree_key]
    return classify(subtree, i)


def start_building_tree(inputs: List[Any], split_attrs: List[str], target_attr: str) -> DecisionTree:
    label_counts = Counter(getattr(i, target_attr) for i in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    if len(label_counts) == 1:
        return Leaf(most_common_label)

    if not split_attrs:
        return Leaf(most_common_label)

    def split_entropy(attr: str) -> float:
        return partition_entropy_by(inputs, attr, target_attr)

    best_attr = min(split_attrs, key=split_entropy)
    partitions = partition_by(inputs, best_attr)

    new_attrs = [a for a in split_attrs if a != best_attr]

    subtrees = {attr_value: start_building_tree(subset, new_attrs, target_attr) for attr_value, subset in
                partitions.items()}

    return Split(best_attr, subtrees, default_value=most_common_label)


tree = start_building_tree(inputs, ['level', 'lang', 'socialMedia', 'education'], 'did_well')