#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute one hierarchy-consistent confidence for every LLM candidate."""

import math

from build_llm_candidate_tasks import source_task_aliases


def normalize_confidence(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def compute_hierarchy_scores(groups, predictions):
    """Return final scores and an audit summary.

    Each LLM prediction is the direct score of the complete candidate tree.
    The program recursively clamps a parent to the minimum final score of its
    direct children.  A child can be correct while its parent is incorrect,
    but a parent cannot remain more confident than a required child.
    """
    raw_by_task = {}
    duplicate_tasks = []
    bad_items = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            bad_items.append(index)
            continue
        task_id = str(item.get("task_id", "")).strip()
        if not task_id:
            bad_items.append(index)
            continue
        if task_id in raw_by_task:
            duplicate_tasks.append(task_id)
        raw_by_task[task_id] = normalize_confidence(item.get("confidence", 0.0))

    candidates_by_node = {}
    task_to_node = {}
    for group in groups:
        for candidate in group.get("candidates", []):
            node_id = str(candidate["node_id"])
            task_id = str(candidate["task_id"])
            if node_id in candidates_by_node:
                raise ValueError("duplicate hierarchy node_id: {}".format(node_id))
            if task_id in task_to_node:
                raise ValueError("duplicate hierarchy task_id: {}".format(task_id))
            candidates_by_node[node_id] = candidate
            task_to_node[task_id] = node_id

    expected_tasks = set(task_to_node)
    missing_tasks = sorted(expected_tasks - set(raw_by_task))
    extra_tasks = sorted(set(raw_by_task) - expected_tasks)

    state = {}
    final_by_node = {}
    adjusted_by_layer = {}

    def resolve(node_id):
        node_state = state.get(node_id, 0)
        if node_state == 1:
            raise ValueError("cycle in candidate hierarchy at node {}".format(node_id))
        if node_state == 2:
            return final_by_node[node_id]
        if node_id not in candidates_by_node:
            raise ValueError("candidate hierarchy references absent child {}".format(node_id))

        state[node_id] = 1
        candidate = candidates_by_node[node_id]
        task_id = str(candidate["task_id"])
        raw_score = raw_by_task.get(task_id, 0.0)
        child_scores = [resolve(child_id) for child_id in candidate.get("direct_children", [])]

        # L1 has no children, so final_score == raw_score.  L2/L3 is the
        # minimum of its own complete-tree score and all child final scores.
        final_score = min([raw_score] + child_scores) if child_scores else raw_score
        final_by_node[node_id] = final_score
        state[node_id] = 2

        if final_score < raw_score:
            layer = str(candidate.get("layer", "unknown"))
            adjusted_by_layer[layer] = adjusted_by_layer.get(layer, 0) + 1
        return final_score

    final_by_task = {}
    for task_id, node_id in task_to_node.items():
        final_by_task[task_id] = resolve(node_id)

    summary = {
        "prediction_items": len(predictions),
        "expected_tasks": len(expected_tasks),
        "scored_tasks": len(set(raw_by_task) & expected_tasks),
        "missing_tasks": missing_tasks,
        "extra_tasks": extra_tasks,
        "duplicate_tasks": duplicate_tasks,
        "bad_items": bad_items,
        "hierarchy_adjusted": sum(adjusted_by_layer.values()),
        "hierarchy_adjusted_by_layer": adjusted_by_layer,
    }
    return final_by_task, summary


def apply_final_scores(rows, groups, final_by_task):
    """Write final scores to every original candidate, including aliases."""
    aliases = source_task_aliases(groups)
    total = 0
    missing_source_tasks = []
    for row_id, row in enumerate(rows):
        for triple_id, triple in enumerate(row.get("triples", [])):
            total += 1
            source_task_id = "{}:{}".format(row_id, triple_id)
            scoring_task_id = aliases.get(source_task_id)
            if scoring_task_id is None or scoring_task_id not in final_by_task:
                missing_source_tasks.append(source_task_id)
                triple["confidence"] = 0.0
            else:
                triple["confidence"] = final_by_task[scoring_task_id]
    return total, missing_source_tasks
