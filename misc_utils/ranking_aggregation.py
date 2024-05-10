from typing import Iterable

import numpy as np
from tqdm.auto import trange


# slow but simple implementation for validation
def _common_x_top_y_ref(x: int, *rankings: Iterable):
    assert 1 <= x <= len(rankings)
    ranked_lists = [list(r) for r in rankings]

    result = {}
    for i in trange(1, max(len(r) for r in ranked_lists) + 1):
        elements, counts = np.unique(sum([r[:i] for r in ranked_lists], []), return_counts=True)
        els_i = elements[counts >= x]
        for e in els_i:
            if e not in result:
                result[e] = i
    return result

def common_x_top_y(x: int, *rankings: Iterable):
    assert 1 <= x <= len(rankings)
    ranked_lists = [list(r) for r in rankings]

    elements = np.array(list(set(sum(ranked_lists, []))))
    element_map = {e: i for i, e in enumerate(elements)}
    counts = np.zeros(shape=len(element_map), dtype=int)
    unseen_mask = np.ones(shape=len(element_map), dtype=bool)
    result = {}
    for i in trange(max(len(r) for r in ranked_lists)):
        for r in ranked_lists:
            if len(r) > i:
                counts[element_map[r[i]]] += 1
        best_el_ids = (counts >= x) & unseen_mask
        (current_ids,) = np.where(best_el_ids)
        for ii in current_ids:
            unseen_mask[ii] = False
            result[elements[ii]] = i + 1

    return result

