from typing import Annotated, Literal

from langchain_core.tools import tool, InjectedToolArg

from .utils import tavily_search_multiple, deduplicate_search_results, process_search_results, format_search_output

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topics: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """
    Fetch results from Tavily Search API with content summarization

    Args: 
        query: The search query string to execute against the Tavily API.
        max_results: Maximum number of search results to retrieve.
        topics: Topic filter for search results, can be "general", "news", or "finance".

    Returns:
        A list of dictionaries containing the search results from the Tavily API.
    """
    search_results = tavily_search_multiple(
        [query], max_results=max_results, topic=topics, include_raw_content=True
    )
    
    unique_results = deduplicate_search_results(search_results)
    
    summarized_results = process_search_results(unique_results)
    
    return format_search_output(summarized_results)


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"