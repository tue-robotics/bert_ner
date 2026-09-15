import logging
import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.pth"
VOCAB_FILENAME = "vocab.slot"

DATA_DIR_ENV_VAR = "NER_MODEL_DATA_DIR"

# Data shipped with the source (vocab.slot). resolve() so a symlinked
# devel-space module still finds the source tree.
PACKAGE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# The weights are ~433 MB and deliberately not distributed with the source, so
# also look in the shared location other tue-robotics packages use for models.
SHARED_DATA_DIR = Path.home() / "data" / "ner_model"

DOWNLOAD_HINT = "see docs/teammate_setup.md for where to download it"


def data_dir_candidates():
    """Directories searched for model data, highest precedence first."""
    candidates = []
    env_dir = os.environ.get(DATA_DIR_ENV_VAR)
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(SHARED_DATA_DIR)
    candidates.append(PACKAGE_DATA_DIR)
    return candidates


def find_data_file(filename, explicit_path=None):
    """
    Locate a model data file, preferring an explicitly configured location.

    Precedence: `explicit_path`, then ``$NER_MODEL_DATA_DIR``, then
    ``~/data/ner_model``, then the data directory shipped with the package.

    :param filename: name of the file to look for, e.g. ``model.pth``
    :param explicit_path: a file, or a directory expected to contain `filename`
    :return: (Path) path to the existing file
    :raises FileNotFoundError: with the searched locations and a download hint
    """
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise FileNotFoundError(
                "{} not found at the configured path '{}'; {}.".format(filename, path, DOWNLOAD_HINT)
            )
        return path

    searched = []
    for directory in data_dir_candidates():
        path = directory / filename
        searched.append(str(path))
        if path.is_file():
            return path

    raise FileNotFoundError(
        "{} not found. Searched: {}. It is not distributed with the source, so it must be "
        "installed manually; {}. Override the location with ${} or, under ROS, the "
        "'ner_model/model_path' parameter.".format(
            filename, ", ".join(searched), DOWNLOAD_HINT, DATA_DIR_ENV_VAR
        )
    )


def _load_slot_names():
    vocab_path = find_data_file(VOCAB_FILENAME)
    names = ["[PAD]"]
    names += vocab_path.read_text("utf-8").strip().splitlines()
    return names


class JointIntentAndSlotFillingModel(nn.Module):
    slot_names = _load_slot_names()
    slot_map = {label: i for i, label in enumerate(slot_names)}

    def __init__(self, slot_num_labels, model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path=None, base_model_name="bert-base-cased"):
    """
    Load the slot tagging model and its tokenizer.

    :param model_path: optional path to the weights, either the file itself or a
        directory containing `model.pth`. When omitted, the locations described
        in :func:`find_data_file` are searched.
    :param base_model_name: pretrained transformer to build the tagger on. It is
        fetched from the Hugging Face cache, which needs populating on first use.
    :return: (model, tokenizer, device)
    """
    device = get_device()

    # Resolve the weights before building the model, so a missing file fails
    # immediately instead of after fetching the pretrained transformer.
    weights_path = find_data_file(MODEL_FILENAME, explicit_path=model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    slot_num_labels = len(JointIntentAndSlotFillingModel.slot_map)
    model = JointIntentAndSlotFillingModel(slot_num_labels=slot_num_labels,
                                          model_name=base_model_name)

    logger.info("Loading NER weights from %s", weights_path)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)

    return model, tokenizer, device
