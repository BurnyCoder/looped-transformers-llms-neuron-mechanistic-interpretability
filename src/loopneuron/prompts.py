"""LLM prompt templates, copied verbatim from established auto-interp / steering-eval work (both Apache-2.0).

- EXPLAINER_SYSTEM and DETECTION_* : EleutherAI Delphi (Paulo et al. 2024, https://arxiv.org/abs/2410.13928),
  files delphi/explainers/default/prompts.py and delphi/scorers/classifier/prompts/detection_prompt.py
  at https://github.com/EleutherAI/delphi .
- JUDGE_CONCEPT / JUDGE_FLUENCY : AxBench (Wu et al. 2025, https://arxiv.org/abs/2501.17148),
  axbench/evaluators/prompt_templates.py at https://github.com/stanfordnlp/axbench .
Global context: autointerp.py formats these; llm.py sends them to the local Qwen3.5 server and logs them in full.
"""

EXPLAINER_SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""

DETECTION_SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain latent of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples possess the latent.

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""

DETECTION_FEWSHOT = [  # Delphi's three few-shot examples (Neuronpedia GPT-2 small latents)
    ("""Latent explanation: Words related to American football positions, specifically the tight end position.

Test examples:

Example 0:<|endoftext|>Getty Images

Patriots tight end Rob Gronkowski had his boss's
Example 1: names of months used in The Lord of the Rings:

"...the
Example 2: Media Day 2015

LSU defensive end Isaiah Washington (94) speaks to the
Example 3: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 4: line, with the left side -- namely tackle Byron Bell at tackle and guard Amini
""", "[1,0,0,0,1]"),
    ("""Latent explanation: The word "guys" in the phrase "you guys".

Test examples:

Example 0: enact an individual health insurance mandate?", Pelosi's response was to dismiss both
Example 1: birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle
Example 2: du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because
Example 3:
<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre
Example 4: the<|endoftext|>"I want to remind you all that 10 days ago (director Massimil
""", "[0,0,0,0,0]"),
    ("""Latent explanation: "of" before words that start with a capital letter.

Test examples:

Example 0: climate, Tomblin's Chief of Staff Charlie Lorensen said.

Example 1: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 2:

Deborah Sathe, Head of Talent Development and Production at Film London,
Example 3:

It has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? · Finally, we believe the
""", "[1,1,1,1,1]"),
]

DETECTION_PROMPT = """Latent explanation: {explanation}

Text examples:

{examples}
"""

JUDGE_CONCEPT = """[System]
Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

JUDGE_FLUENCY = """[System]
Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability. After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""
