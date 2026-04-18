import rospy
from typing import List
from .model import load_model
from .inference import InferenceService
from .mapper import build_semantics


class NERParser:
    _service = None

    def __init__(self):
        if NERParser._service is None:
            rospy.loginfo("Loading NER model")
            model, tokenizer, device = load_model()
            NERParser._service = InferenceService(model, tokenizer, device)
            rospy.loginfo("NER model ready")

    @classmethod
    def fromstring(cls, grammar):
        return cls()

    def parse(self, target, sentence):
        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        results = NERParser._service.predict(sentence)

        rospy.loginfo("NER input:  '%s'", sentence)
        rospy.loginfo("NER output: %s", results)

        semantics = build_semantics(results)
        rospy.loginfo("Mapped semantics: %s", semantics)
        return semantics

    def parse_raw(self, target, words, debug=False):
        return self.parse(target, words)

    def get_random_sentence(self, target=None):
        return "bring me a coke from the kitchen"

    def verify(self, target=None):
        pass
