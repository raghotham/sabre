"""
Web search helper using DuckDuckGo.

Provides free web search without API keys.
"""

import logging
from typing import List

from sabre.common.models import SearchResult

logger = logging.getLogger(__name__)


class Search:
    """Web search helper using DuckDuckGo."""

    @staticmethod
    def web_search(
        query: str,
        total_links_to_return: int = 10,
    ) -> List[SearchResult]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query
            total_links_to_return: Max number of results to return

        Returns:
            List of SearchResult objects with url, title, snippet, engine
        """

        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.error("ddgs not installed. Run: uv pip install ddgs")
                return []

        logger.info(f"Searching web for: {query}")

        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=total_links_to_return))

            return_results: List[SearchResult] = []
            for result in results:
                return_results.append(
                    SearchResult(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        snippet=result.get("body", ""),
                        engine="DuckDuckGo",
                    )
                )

            logger.info(f"Found {len(return_results)} results")

            # Print results to stdout so they're captured in execution output
            if return_results:
                print(f"Search results for '{query}':")
                for i, result in enumerate(return_results, 1):
                    print(f"\n{i}. {result.title}")
                    print(f"   URL: {result.url}")
                    print(f"   {result.snippet[:200]}...")

            return return_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
