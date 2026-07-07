import rospy

from .model import load_model
from .inference import InferenceService
from .mapper import build_semantics
from .candidates import load_candidates
from .linker import BiEncoderLinker


class NERParser:
    _service = None
    _linker = None

    def __init__(self):
        if NERParser._service is None:
            rospy.loginfo("Loading NER model")
            model, tokenizer, device = load_model()
            NERParser._service = InferenceService(model, tokenizer, device)
            rospy.loginfo("NER model ready")

        if NERParser._linker is None:
            NERParser._linker = NERParser._create_linker()

    @staticmethod
    def _create_linker():
        enabled = rospy.get_param("entity_linker/enabled", True)
        if not enabled:
            rospy.loginfo("Entity linker: disabled via param entity_linker/enabled")
            return None

        candidates = load_candidates()
        if not candidates:
            rospy.logwarn(
                "Entity linker: no candidates loaded — entity linking disabled"
            )
            return None

        model_name = rospy.get_param(
            "entity_linker/model_name", "all-MiniLM-L6-v2"
        )
        threshold = rospy.get_param("entity_linker/threshold", 0.6)

        try:
            linker = BiEncoderLinker(
                candidates=candidates,
                model_name=model_name,
                threshold=threshold,
            )
            rospy.loginfo(
                "Entity linker: initialized (model=%s, threshold=%.2f)",
                model_name,
                threshold,
            )
            return linker
        except Exception as exc:
            rospy.logwarn("Entity linker: failed to initialize: %s", exc)
            return None

    @classmethod
    def fromstring(cls, grammar):
        return cls()

    def parse(self, target, sentence):
        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        rospy.loginfo("NER input: '%s'", sentence)
        results = NERParser._service.predict(sentence)
        rospy.loginfo("NER output: %s", results)

        semantics = build_semantics(results, linker=NERParser._linker)
        rospy.loginfo("Mapped semantics: %s", semantics)
        return semantics

    def parse_raw(self, target, words, debug=False):
        return self.parse(target, words)

    def get_random_sentence(self, target=None):
        return "bring me a coke from the kitchen"

    def verify(self, target=None):
        pass
