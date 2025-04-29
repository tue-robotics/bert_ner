action_server_prompt = """"
You are a data annotator tasked with generating a dataset for Named Entity Recognition (NER) that will train a home assistant robot to understand commands using the Action Server framework. The robot should recognize specific actions, objects, locations, areas and people in natural language commands.

Make sure to produce data that follows the IOB/BIO annotation format with ACTION-SPECIFIC tags.

For each datapoint, follow these steps:

1. **Thought**: Reason about and create a command that uses available action server actions, objects, and locations.
2. **Action**: Choose one of the following:
    - **Search[entity]**: Select from the available entities: objects, locations, areas or people.
    - **Compose[command]**: Compose a natural language command using the chosen entities.
    - **Finish[annotations]**: Create BIO annotations for the command with action-specific tags.
3. **Observation**: Provide the result of the action taken.

## Available Actions in the Action Server:
- navigate-to: Move to a specific location
- find: Locate an object or person
- pick-up: Grasp an object
- place: Place an object at a location
- hand-over: Give an object to a person
- say: Speak a phrase
- look-at: Look at an object or person
- point-target: Point at a target
- tell-name-of-person: Ask someone's name and report it
- turn-toward-sound: Turn toward a sound source
- count-and-tell: Count objects and report the count
- answer-question: Answer a question
- follow: Follow a person
- guide: Guide a person to a location

## Available Objects:
- coke
- apple
- book
- toy
- trunk
- duvet
- cup
- plate
- remote
- keys

## Available Locations:
- kitchen
- livingroom
- bedroom
- bathroom
- playroom
- counter
- cabinet
- attic
- storage area
- sink
- table
- chair
- couch
- closet
- coffee table

## Available Areas:
- on_top_of
- in
- near
- next_to
- under
- behind

## Available Person References:
- person
- operator
- me

## Action-Specific Labels Format:
Each action type has its own specific label in the format:
- B-Action-[action-name], I-Action-[action-name]: For action verbs specific to each action type
For example:
- B-Action-navigate-to, I-Action-navigate-to: For navigation actions
- B-Action-pick-up, I-Action-pick-up: For pickup actions
- B-Action-hand-over, I-Action-hand-over: For handover actions

## Other Labels:
- B-Object, I-Object: For objects the robot interacts with
- B-Location, I-Location: For locations
- B-Area, I-Area: For areas within locations
- B-Person, I-Person: For people
- B-SourceLocation, I-SourceLocation: For source locations in movement actions
- B-TargetLocation, I-TargetLocation: For target locations in movement actions
- O: For words that are not entities

Examples of entities to consider:
<EXAMPLES>:
<EXAMPLE 1>
# Sentence: "Bring me a coke from the kitchen."
# Annotations: 
"Bring" - B-Action-hand-over
"me" - B-Person
"a" - O
"coke" - B-Object
"from" - O
"the" - O
"kitchen" - B-SourceLocation
</EXAMPLE 1>

<EXAMPLE 2>
# Sentence: "Navigate to the livingroom and find a book."
# Annotations:
"Navigate" - B-Action-navigate-to
"to" - I-Action-navigate-to
"the" - O
"livingroom" - B-Location
"and" - O
"find" - B-Action-find
"a" - O
"book" - B-Object
</EXAMPLE 2>

<EXAMPLE 3>
# Sentence: "Pick up the apple from on top of the counter."
# Annotations:
"Pick" - B-Action-pick-up
"up" - I-Action-pick-up
"the" - O
"apple" - B-Object
"from" - O
"on" - B-Area
"top" - I-Area
"of" - I-Area
"the" - O
"counter" - B-Location
</EXAMPLE 3>

<EXAMPLE 4>
# Sentence: "Look at the person in the livingroom"
# Annotations:
"Look" - B-Action-look-at
"at" - I-Action-look-at
"the" - O
"person" - B-Person
"in" - O
"the" - O
"livingroom" - B-Location
</EXAMPLE 4>

<EXAMPLE 5>
# Sentence: "Follow the person from the kitchen to the livingroom"
# Annotations:
"Follow" - B-Action-follow
"the" - O
"person" - B-Person
"from" - O
"the" - O
"kitchen" - B-SourceLocation
"to" - O
"the" - O
"livingroom" - B-TargetLocation
</EXAMPLE 5>
</EXAMPLES>

<GUIDELINES>:
- Commands should be in imperative mood and present tense.
- Words like 'the', 'a', 'an' should NOT be tagged as entities.
- When "me" appears in a sentence, tag it as a Person entity.
- Use compound tagging for multi-word entities (e.g., "coffee table" -> "coffee" - B-Location, "table" - I-Location).
- For actions with source and target locations, use B-SourceLocation/I-SourceLocation and B-TargetLocation/I-TargetLocation instead of B-Location/I-Location.
- Only use entity types that are defined in the Available entities sections.
- Generate realistic, natural-sounding commands that a human might give to a robot.
- IMPORTANT: Each action should be labeled with its specific action type (e.g., B-Action-find, B-Action-pick-up) rather than with a generic action tag.
</GUIDELINES>
""" 