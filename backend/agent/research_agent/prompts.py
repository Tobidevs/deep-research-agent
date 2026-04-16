RESEARCH_AGENT_SYSTEM_PROMPT = """Use tavily_search and think_tool to research the given topic.
After each search, call think_tool to assess what you found, what's still missing, and whether you have enough to answer. Start with broad queries, then narrow.
Search budget: 1-2 calls for simple to complex questions, 3 calls maximum. Stop as soon as you can answer comprehensively."""


SUMMARIZE_WEBPAGE_PROMPT = """Summarize the following webpage for a downstream research agent. Target 25–30% of the original length. 
Preserve the main topic, key facts, statistics, important quotes, dates, and conclusions.

Today's date is {date}

<webpage_content>
{webpage_content}
</webpage_content>
"""

COMPRESS_RESEARCH_SYSTEM_PROMPT = """Clean up raw research findings from tool calls and web searches into a structured report. Preserve all substantive 
information verbatim — deduplicate only (e.g. "three sources all stated X"). Skip think_tool calls entirely; include only tavily_search results and web findings.

<Output Format>
**Queries and Tool Calls Made**
**Comprehensive Findings** (all sources, inline citations [1], [2]...)
**Sources**
[1] Title: URL
[2] Title: URL
</Output Format>

Preserving every source is critical — a downstream LLM will merge this report with others."""

COMPRESS_RESEARCH_REMINDER_PROMPT = """Using the research messages above, produce the compressed report now. 
Remember: preserve all tavily_search findings verbatim, skip think_tool calls, and include every source with inline citations."""

FINAL_REPORT_PROMPT = """Using the research findings above, write a comprehensive report on:
RESEARCH TOPIC: {research_brief}

Choose a report structure appropriate for the topic. Use inline citations [1], [2]... throughout. Do not introduce any information not present in the research above."""
