from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import nest_asyncio
import asyncio

# List of URLs to crawl
urls_to_crawl = [
    "https://docs.llamaindex.ai/en/stable/understanding/",
]

# Synchronous wrapper
def crawl_sync():
    async def crawl_with_crawl4ai():
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, # Bypass cache to always fetch fresh content
            page_timeout=80000, # Set a high timeout for page loading
            word_count_threshold=50 # Minimum word count to consider a page valid
        )

        data_res = {"data": []}

        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun_many(
                urls_to_crawl,
                config=config
            )

            for result in results:
                if result.success:
                    title = result.metadata.get("title", "")
                    if not title and result.markdown: # Fallback to extracting title from markdown
                        lines = result.markdown.raw_markdown.split('\n')
                        for line in lines: 
                            if line.startswith('#'):
                                title = line.strip('#').strip()
                                break

                    data_res["data"].append({ 
                        "text": result.markdown.raw_markdown if result.markdown else "",
                        "meta": {
                            "url": result.url,
                            "meta": {
                                "title": title
                            }
                        }
                    })

        return data_res

    # Handle async execution
    nest_asyncio.apply()

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(crawl_with_crawl4ai())
    loop.close()
    return result

# Run the crawler
data_res = crawl_sync()

# Print results (same format as before)
print("URL:", data_res["data"][0]["meta"]["url"])
print("Title:", data_res["data"][0]["meta"]["meta"]["title"])
print("Content:", data_res["data"][0]["text"][0:500], "...")