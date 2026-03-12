CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT = """Today is {date}.

<messages>
{messages}
</messages>

Determine if you need to ask a clarifying question before starting research.

Rules:
- If you have already asked a clarifying question in the messages above, do NOT ask another unless absolutely necessary
- Ask for clarification if the request is ambiguous, incomplete, or contains unknown acronyms/terms
- Never ask for information the user has already provided

If asking a question, be concise and gather everything needed in one message. Use markdown bullet points if listing multiple questions.
"""

TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT = """Today is {date}.

<messages>
{messages}
</messages>

Translate the conversation above into a detailed research brief written in the first person from the user's perspective.

Guidelines:
- Include every detail, preference, and constraint the user explicitly stated
- For dimensions the user didn't specify, note them as open (e.g. "no price range specified — consider all options") rather than assuming
- Never invent preferences or narrow scope beyond what the user stated
- Distinguish between user preferences (strict — only what was stated) and research scope (can be broader to ensure comprehensive coverage)

Source preferences:
- Products/travel: prefer official brand sites or primary sources over aggregators
- Academic: link to original papers or journals, not summaries
- People: prefer LinkedIn or personal websites
- Match source language to the language of the query
"""