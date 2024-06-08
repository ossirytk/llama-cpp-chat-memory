### Card Format
See [character editor](https://character-tools.srjuggernaut.dev/character-editor).<BR>
There are few example cards included like'Skynet' and 'Shodan'<BR>
'name' : 'char_name'<br>
The name for the ai character. When using json or yaml, this is expected to correspond to avatar image. name.png or name.jpg.<br>
'description' : 'char_persona'<br>
The description for the character personality. Likes, dislikes, personality traits.<br>
'scenario' : 'world_scenario'<br>
Description of the scenario. This roughly corresponds to things like "You are a hr customer service having a discussion with a customer. Always be polite". etc.<br>
'mes_example' : 'example_dialogue'<br>
Example dialogue. The AI will pick answer patterns based on this<br>
'first_mes' : 'char_greeting'<br>
A landing page for the chat. This will not be included in the prompt.

The documents folder includes some documents for embeddings parsing for the character cards.