from typing import (
    Generator, Iterable, List, Literal, Optional, Tuple, Type, Union
)

from tqdm.auto import tqdm
from joblib.memory import Memory
from aiida.tools.graph.graph_traversers import traverse_graph
from aiida.common import LinkType
from aiida.orm import Node, QueryBuilder, CalcJobNode

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

def _get_node_by_pk(pk: int, assert_type: Optional[Type[Node]] = None) -> Node:
    (node,) = QueryBuilder().append(Node, filters={"id": pk}).all(flat=True)
    if assert_type is not None:
        assert isinstance(node, assert_type)
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

@memory.cache(ignore=["verbose"])
def _find_creator_calc_job_nodes(
    pks: List[int],
    verbose: bool,
) -> List[List[int]]:
    def _recurse_find_creators(node: Node) -> Generator[Node, None, None]:
        if node.creator is None:
            return
        if isinstance(node.creator, CalcJobNode):
            yield node.creator
        else:
            for key in node.creator.inputs:
                for creator in _recurse_find_creators(node.creator.inputs[key]):
                    yield creator

    progress = tqdm if verbose else (
        lambda x, *argv, **kwargs: x
    )

    creators = []
    for pk in progress(pks, desc="Searching for creator calc jobs"):
        (node,) = QueryBuilder().append(Node, filters=dict(id=pk)).all(flat=True)
        creators.append([
            c.pk for c in _recurse_find_creators(node)
        ])
    return creators

def find_creator_calc_job_nodes(
    nodes: Iterable[Union[Node, int]],
    verbose: bool = True,
) -> List[List[CalcJobNode]]:
    progress = tqdm if verbose else (
        lambda x, *argv, **kwargs: x
    )

    pks = [n if isinstance(n, int) else n.pk for n in nodes]
    creator_pks = _find_creator_calc_job_nodes(pks, verbose)

    return [
        [_get_node_by_pk(pk, CalcJobNode) for pk in pk_set]
        for pk_set in progress(creator_pks, desc="Getting calc job nodes by pk")
    ]
