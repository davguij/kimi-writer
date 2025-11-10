"""
Web search tool implementation
"""
from duckduckgo_search import DDGS

def web_search_impl(query: str) -> str:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query: The search query.

    Returns:
        A string containing the search results, formatted for display.
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return "\n".join([f"[{i+1}] \"{r['title']}\"\n{r['body']}\nURL: {r['href']}" for i, r in enumerate(results)])
