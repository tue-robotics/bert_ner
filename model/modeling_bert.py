from torch import nn
from transformers import AutoModel

class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, slot_num_labels, model_name="bert-base-cased", dropout_prob=0.1):
        super(JointIntentAndSlotFillingModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.dropout = nn.Dropout(dropout_prob)

        # Intent and slot classifiers
        # self.intent_classifier = nn.Linear(
        # self.bert.config.hidden_size, intent_num_labels)
        self.slot_classifier = nn.Linear(
            self.bert.config.hidden_size, slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT outputs (sequence and pooled)
        # In the forward method:
        # print("########################################### Forward_pass:")
        # print("FORWARD input_ids:", input_ids.shape)
        # print("FORWARD attention_mask:", attention_mask.shape)
        # print("FORWARD Max token ID:", torch.max(input_ids))

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # pooled_output = outputs.pooler_output

        # Slot filling using sequence_output
        sequence_output = self.dropout(sequence_output)
        slot_logits1 = self.slot_classifier(sequence_output)

        # Intent classification using pooled_output
        # pooled_output = self.dropout(pooled_output)
        # intent_logits = self.intent_classifier(pooled_output)

        return slot_logits1  # , intent_logits