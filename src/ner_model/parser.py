import rospy
from typing import List
from .model import load_model
from .inference import InferenceService


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

    def parse(self, sentence: List[str]):
        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        results = NERParser._service.predict(sentence)

        rospy.loginfo("NER input:  '%s'", sentence)
        rospy.loginfo("NER output: %s", results)

        return self._hardcoded_semantics(sentence)

    def parse_raw(self, target, words, debug=False):
        return self.parse(target, words)

    def get_random_sentence(self, target=None):
        return "bring me a coke from the kitchen"

    def verify(self, target=None):
        pass

    def _hardcoded_semantics(self, sentence):
        return {
            "actions": [{
                "action": "bring",
                "object": {"type": "coke"},
                "target-location": {"type": "person", "id": "operator"},
            }]
        }
