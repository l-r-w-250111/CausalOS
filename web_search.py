from ddgs import DDGS
import json

def perform_web_search(query: str, max_results: int = 5):
    """
    Performs a web search using DuckDuckGo and returns a formatted string of results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        str: A formatted string containing the search results with titles,
             snippets, and source URLs, ready to be used as context for an LLM.
             Returns an error message if the search fails or no results are found.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No search results found."

        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = (
                f"Result {i+1}:\n"
                f"  Title: {result.get('title', 'N/A')}\n"
                f"  Snippet: {result.get('body', 'N/A')}\n"
                f"  Source: {result.get('href', 'N/A')}\n"
            )
            formatted_results.append(formatted_result)

        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"An error occurred during web search: {e}"

if __name__ == '__main__':
    # Example usage for testing
    test_query = "latest news on generative AI"
    print(f"Performing search for: '{test_query}'")
    search_results = perform_web_search(test_query)
    print("\n--- Search Results ---")
    print(search_results)
    print("\n--- End of Search Results ---\n")

    test_query_obama = "Obama"
    print(f"Performing search for: '{test_query_obama}'")
    search_results_obama = perform_web_search(test_query_obama)
    print("\n--- Search Results ---")
    print(search_results_obama)
    print("\n--- End of Search Results ---\n")


