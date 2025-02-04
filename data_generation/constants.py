cot_ner_prompt =  """"
You are a data annotator tasked with generating a dataset for a Named Entity Recognition (NER) task that will be used for 
training a mobile home assistant robot that will be able to perform basic home assistance tasks 
such as grabbing and moving objects. The robot through an interface should be able to receive commands, and use the NER model
to recognize entities and their position in the house. 
Make sure to produce data that follows the IOB/BIO annotation format.
For each datapoint, perform the following steps:

1. **Thought**: Reason and create sentence to identify potential entities and locations.
2. **Action**: Decide on an action to identify entities, which can be one of the following:
    - **Search[entity]**: Think of potential items that a home robot should be interacting with and their locations within a household.
    - **Compose[keyword]**: Think of how the robot should be expected to be commanded by a human on how to interact with these items and their location.
    - **Finish[answer]**: Create the text and the accompanying annotations.
3. **Observation**: Provide the result of the action taken.

Repeat these steps until all entities in the sentence are created and labeled.

<GUIDELINES>:
** Since the robot is expected to receive commands, the sentences should be in the imperative mood and in the present tense.
Example: "Move the cup from  the bathroom to the kitchen." or "Grab the book from the living room." 
</GUIDELINES>
"""

