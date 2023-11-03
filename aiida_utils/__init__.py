from typing import Iterable, List, Literal, Tuple, Union

from tqdm.auto import tqdm
from joblib.memory import Memory
from aiida.tools.graph.graph_traversers import traverse_graph
from aiida.common import LinkType
from aiida.orm import Node, QueryBuilder

memory = Memory("cache/", verbose=0)


@memory.cache(ignore=["verbose"])
def _find_connections(
    pks1: List[int], pks2: List[int], left_to_right: bool, verbose: bool,
) -> List[Tuple[List[int], List[int]]]:
    progress = tqdm if verbose else (
        lambda x, *argv, **kwargs: x
    )
    link_types = [
        LinkType.CREATE, LinkType.RETURN, LinkType.INPUT_CALC,
        LinkType.INPUT_WORK, LinkType.CALL_CALC, LinkType.CALL_WORK,
    ]

    kwargs = dict(
        links_forward=link_types,
    ) if left_to_right else dict(
        links_backward=link_types,
    )
    travs = [
        traverse_graph([pk], **kwargs)
        for pk in progress(pks1, desc="Traversing graph")
    ]
    return [
        ([pk], [pk2 for pk2 in pks2 if pk2 in trav["nodes"]])
        for pk, trav in zip(pks1, travs)
    ]

def _get_node_by_pk(pk: int) -> Node:
    (node,) = QueryBuilder().append(Node, filters={"id": pk}).all(flat=True)
    return node

def find_connections(
    nodes_from: Iterable[Union[Node, int]],
    nodes_to: Iterable[Union[Node, int]],
    mode: Literal["from_left", "to_right"],
    verbose: bool = True,
) -> List[Tuple[List[Node], List[Node]]]:
    pks_from = [n if isinstance(n, int) else n.pk for n in nodes_from]
    pks_to = [n if isinstance(n, int) else n.pk for n in nodes_to]


    if mode == "from_left":
        results = _find_connections(pks_from, pks_to, left_to_right=True, verbose=verbose)
    elif mode == "to_right":
        results = [
            (ll, rr) for rr, ll in _find_connections(pks_to, pks_from, left_to_right=False, verbose=verbose)
        ]
    else:
        raise NotImplementedError(mode)

    return [
        (
            [_get_node_by_pk(n) for n in ll],
            [_get_node_by_pk(n) for n in rr],
        ) for ll, rr in results
    ]
