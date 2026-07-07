from dataclasses import dataclass
from typing import List, Optional, Sequence

import rospy

SLOT_TO_CANDIDATE_LABEL = {
    "Object": "Object",
    "SourceLocation": "Location",
    "TargetLocation": "Location",
    "Location": "Location",
    "Person": "Person",
    "Area": "Location",
}


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    label: str
    text: str


@dataclass(frozen=True)
class LinkResult:
    entity_id: str
    score: float
    mention: str
    slot_label: str
    matched_text: str


class BiEncoderLinker:
    """Zero-shot entity linker using sentence-transformer cosine similarity."""

    def __init__(
        self,
        candidates: Sequence[EntityCandidate],
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.6,
        top_k_log: int = 3,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for entity linking. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._threshold = threshold
        self._top_k_log = top_k_log
        self._candidates = list(candidates)
        self._candidates_by_label = {}
        for candidate in self._candidates:
            self._candidates_by_label.setdefault(candidate.label, []).append(candidate)

        rospy.loginfo(
            "Entity linker: loading model '%s' with %d candidates (%s)",
            model_name,
            len(self._candidates),
            ", ".join(
                "{}={}".format(label, len(items))
                for label, items in sorted(self._candidates_by_label.items())
            ),
        )
        self._model = SentenceTransformer(model_name)
        self._embeddings_by_label = {}
        for label, label_candidates in self._candidates_by_label.items():
            texts = [candidate.text for candidate in label_candidates]
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            self._embeddings_by_label[label] = (label_candidates, embeddings)
            rospy.logdebug(
                "Entity linker: precomputed %d embeddings for label '%s'",
                len(label_candidates),
                label,
            )
        rospy.loginfo("Entity linker: ready (threshold=%.2f)", self._threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    def _exact_match(
        self, mention: str, label_candidates: List[EntityCandidate]
    ) -> Optional[LinkResult]:
        normalized = mention.lower().strip().replace(" ", "_")
        normalized_mention = mention.lower().strip()
        for candidate in label_candidates:
            if (
                candidate.entity_id == normalized
                or candidate.text.lower() == normalized_mention
                or candidate.entity_id.replace("_", " ") == normalized_mention
            ):
                return LinkResult(
                    entity_id=candidate.entity_id,
                    score=1.0,
                    mention=mention,
                    slot_label="",
                    matched_text=candidate.text,
                )
        return None

    def link(self, slot_label: str, mention: str) -> Optional[LinkResult]:
        candidate_label = SLOT_TO_CANDIDATE_LABEL.get(slot_label)
        if candidate_label is None:
            rospy.logwarn(
                "Entity linker: unknown slot '%s' for mention '%s'", slot_label, mention
            )
            return None

        label_candidates, embeddings = self._embeddings_by_label.get(
            candidate_label, ([], None)
        )
        if not label_candidates:
            rospy.logwarn(
                "Entity linker: no candidates for slot '%s' (label '%s'), mention '%s'",
                slot_label,
                candidate_label,
                mention,
            )
            return None

        exact = self._exact_match(mention, label_candidates)
        if exact is not None:
            result = LinkResult(
                entity_id=exact.entity_id,
                score=exact.score,
                mention=mention,
                slot_label=slot_label,
                matched_text=exact.matched_text,
            )
            rospy.loginfo(
                "Entity linker: exact match '%s' (%s) -> '%s'",
                mention,
                slot_label,
                result.entity_id,
            )
            return result

        mention_embedding = self._model.encode(mention, normalize_embeddings=True)
        scores = mention_embedding @ embeddings.T
        ranked_indices = scores.argsort()[::-1]

        top_k = min(self._top_k_log, len(ranked_indices))
        top_scores = [
            (label_candidates[idx].entity_id, float(scores[idx]))
            for idx in ranked_indices[:top_k]
        ]
        rospy.loginfo(
            "Entity linker: scores for '%s' (%s): %s",
            mention,
            slot_label,
            ", ".join(
                "{}={:.3f}".format(entity_id, score) for entity_id, score in top_scores
            ),
        )

        best_idx = ranked_indices[0]
        best_candidate = label_candidates[best_idx]
        best_score = float(scores[best_idx])

        if best_score < self._threshold:
            rospy.logwarn(
                "Entity linker: rejected '%s' (%s) — best '%s' score %.3f < threshold %.3f",
                mention,
                slot_label,
                best_candidate.entity_id,
                best_score,
                self._threshold,
            )
            return None

        result = LinkResult(
            entity_id=best_candidate.entity_id,
            score=best_score,
            mention=mention,
            slot_label=slot_label,
            matched_text=best_candidate.text,
        )
        rospy.loginfo(
            "Entity linker: linked '%s' (%s) -> '%s' (score=%.3f, matched='%s')",
            mention,
            slot_label,
            result.entity_id,
            result.score,
            result.matched_text,
        )
        return result
