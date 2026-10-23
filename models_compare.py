# -*- coding: utf-8 -*-
"""Thin compare model plus read-only classifier-score capture helpers.

The actual compare policy lives in ``models.ITransformer.predict_scratch`` so
that compare follows the supervised model's original single-pass, layer-wise
candidate generation exactly.  This subclass only keeps the capture utility
used by offline score-distribution audits.
"""

from __future__ import print_function

import copy

import torch as t

import models as baseline_models
from models import ITransformer as BaselineITransformer


RELATION_PREFIX = "RELATION"
ENTITY_PREFIX = "ENTITY"


def _relation_structure(candidate, structures):
    relation, operands = candidate
    op1, op2 = int(operands[0]), int(operands[1])
    if op1 >= len(structures) or op2 >= len(structures):
        raise ValueError(
            "candidate operand exceeds resolved structures: {} / {}".format(
                candidate, len(structures)
            )
        )
    return (
        RELATION_PREFIX,
        relation,
        structures[op1],
        structures[op2],
    )


def _resolve_result(result, entity_count):
    structures = [(ENTITY_PREFIX, index) for index in range(entity_count)]
    relations = []
    for candidate in result:
        structure = _relation_structure(candidate, structures)
        structures.append(structure)
        relations.append(structure)
    return structures, relations


def _score_map_from_capture(captured_candidates, captured_outputs, structures):
    if len(captured_outputs) != 2 * len(captured_candidates):
        raise RuntimeError(
            "unexpected classifier/candidate capture counts: {} vs {}".format(
                len(captured_outputs), len(captured_candidates)
            )
        )

    score_map = {}
    layer_records = []
    for layer_index, candidates in enumerate(captured_candidates):
        first = captured_outputs[2 * layer_index]
        second = captured_outputs[2 * layer_index + 1]
        if first.shape != second.shape or not t.allclose(
            first, second, atol=0.0, rtol=0.0
        ):
            raise RuntimeError("baseline classifier was not stable within one layer")
        if first.shape[0] != 1:
            raise ValueError("score capture currently requires batch_size=1")

        scores = first[0]
        for index, candidate in enumerate(candidates):
            structure = _relation_structure(candidate, structures)
            score = float(scores[index].item())
            previous = score_map.get(structure)
            if previous is None or score > previous:
                score_map[structure] = score
            layer_records.append({
                "layer": layer_index + 1,
                "structure": structure,
                "model_score": score,
            })
    return score_map, layer_records


class ITransformerCompare(BaselineITransformer):
    """Baseline model with the same direct layer-wise compare implementation."""

    def _predict_and_capture(
        self,
        x,
        x_mask,
        entity_index,
        entity_type,
        relations,
        threshold,
        pllm_scores,
        fusion_mode,
        llm_threshold,
        max_layer=6,
    ):
        if x.shape[0] != 1:
            raise ValueError("score capture requires --batch-size 1")

        captured_candidates = []
        captured_outputs = []
        original_generate = baseline_models.generate_candidate2

        def capture_generate(schema, type2index):
            candidates = original_generate(schema, type2index)
            if candidates:
                captured_candidates.append(copy.deepcopy(candidates))
            return candidates

        def capture_classifier(module, inputs, output):
            captured_outputs.append(output.detach().cpu().clone())

        handle = self.cls.register_forward_hook(capture_classifier)
        baseline_models.generate_candidate2 = capture_generate
        try:
            result = super(ITransformerCompare, self).predict_scratch(
                x=x,
                x_mask=x_mask,
                entity_index=entity_index,
                entity_type=entity_type,
                relations=relations,
                threshold=threshold,
                pllm_scores=pllm_scores,
                alpha=1.0,
                fusion_mode=fusion_mode,
                llm_threshold=llm_threshold,
                max_layer=max_layer,
            )
        finally:
            baseline_models.generate_candidate2 = original_generate
            handle.remove()
        return result, captured_candidates, captured_outputs

    def predict_scratch(
        self,
        x,
        x_mask,
        entity_index,
        entity_type,
        relations,
        threshold=0.5,
        pllm_scores=None,
        alpha=1.0,
        fusion_mode="compare",
        llm_threshold=0.0,
        max_layer=6,
    ):
        return super(ITransformerCompare, self).predict_scratch(
            x=x,
            x_mask=x_mask,
            entity_index=entity_index,
            entity_type=entity_type,
            relations=relations,
            threshold=threshold,
            pllm_scores=pllm_scores,
            alpha=alpha,
            fusion_mode=fusion_mode,
            llm_threshold=llm_threshold,
            max_layer=max_layer,
        )
