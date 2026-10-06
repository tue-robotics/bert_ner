import rospy

from .ner_model import load_model
from .inference import InferenceService
from .mapper import build_semantics
from .candidates import load_candidates
from .linker import BiEncoderLinker


class NERParser:
    """
    Drop-in replacement for grammar_parser's CFGParser, backed by the NER model.

    Keeps the CFGParser interface (fromstring, parse, parse_raw, ...) so hmi and
    conversation_engine can switch parsers without further changes. 
    The model and entity linker are loaded once and shared by all instances!
    """

    _service = None
    _linker = None

    def __init__(self):
        if NERParser._service is None:
            rospy.loginfo("Loading NER model")
            model_path = rospy.get_param("ner_model/model_path", None)
            try:
                model, tokenizer, device = load_model(model_path=model_path)
            except FileNotFoundError as exc:
                rospy.logerr("NER model unavailable: %s", exc)
                raise
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
            rospy.logwarn(
                "Entity linker: failed to initialize: %s: %s", type(exc).__name__, exc
            )
            return None

    @classmethod
    def fromstring(cls, grammar):
        return cls()

    def parse(self, target, sentence):
        """
        Turns a spoken command into the semantics dict for the action server.

        1. The NER model tags every token with a slot (an action such as
           "navigate-to", or an entity such as Object / Location / Person).
        2. build_semantics groups the tagged spans into actions, attaching each
           entity to the action before it. Then,using the linker,
           each mention is mapped to a known world-model id
           (e.g. "dinner table" -> "dinner_table").

        e.g. "go to the kitchen and grab a coke" ->
            {"actions": [{"action": "navigate-to", "target-location": {"id": "kitchen"}},
                         {"action": "pick-up", "object": {"type": "coke"}}]}

        :param target: unused, kept for CFGParser compatibility
        :param sentence: the command, as a string or a list of words
        :return: {"actions": [...]}; {"actions": [{"action": "unknown"}]} if
            no action was recognized
        """
        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        rospy.loginfo("NER input: '%s'", sentence)
        try:
            results = NERParser._service.predict(sentence)
        except Exception as exc:
            rospy.logerr(
                "NER model inference failed on '%s': %s: %s",
                sentence,
                type(exc).__name__,
                exc,
            )
            raise
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
