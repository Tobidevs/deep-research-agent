BRIEF_CRITERIA_PROMPT = """You are a research brief evaluator. Determine if the brief adequately captures the specific criterion provided.

<research_brief>
{research_brief}
</research_brief>

<criterion>
{criterion}
</criterion>

CAPTURED if the brief explicitly mentions, paraphrases, or clearly implies the criterion in a way a researcher could act on.
NOT CAPTURED if the criterion is absent, only partially addressed, or contradicted.

Examples:
- Criterion: "Current age is 25" → Brief mentions "25-year-old investor" → CAPTURED
- Criterion: "High risk tolerance" → Brief mentions "willing to accept significant market volatility" → CAPTURED (equivalent concept)
- Criterion: "Monthly rent below 7k" → Brief mentions "apartments in Manhattan" with no budget → NOT CAPTURED
- Criterion: "Doorman building required" → Brief mentions "modern amenities" → NOT CAPTURED

When in doubt about partial coverage, lean NOT CAPTURED."""

BRIEF_HALLUCINATION_PROMPT = """You are a research brief auditor. Determine if the brief introduces any assumptions beyond what the user explicitly stated.

<research_brief>
{research_brief}
</research_brief>

<criterion>
{criterion}
</criterion>

PASS if the brief only includes explicitly stated requirements or logically necessary inferences.
FAIL if the brief adds unstated preferences, demographics, geographic details, or narrows scope beyond what the user specified.

Examples:
- Criterion: "coffee shops in San Francisco" → Brief mentions "trendy spots for young professionals" → FAIL (assumed demographics)
- Criterion: "2 bed under $3000" → Brief mentions "modern apartments in safe neighborhoods" → FAIL (assumed preferences)
- Criterion: "2 bed under $3000" → Brief mentions "2-bedroom apartments within $3000 budget" → PASS

Be strict — if uncertain whether something was user-specified, lean FAIL."""