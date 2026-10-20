import asyncio
from typing import Any


async def fetch_data(url: str, delay: float) -> dict[str, Any]:
    """Simulate an async HTTP request."""
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)
    return {"url": url, "status": 200, "data": f"Response from {url}"}


async def fetch_with_timeout(url: str, timeout: float) -> dict | None:
    """Fetch with timeout — returns None on timeout."""
    try:
        return await asyncio.wait_for(fetch_data(url, 2.0), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Timeout fetching {url}")
        return None


async def fetch_all_concurrent(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs concurrently using gather."""
    tasks = [fetch_data(url, i * 0.3) for i, url in enumerate(urls)]
    return await asyncio.gather(*tasks)


async def producer_consumer():
    """Async producer-consumer pattern with a queue."""
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=5)

    async def producer():
        for i in range(8):
            await queue.put(i)
            print(f"Produced: {i}")
            await asyncio.sleep(0.1)
        await queue.put(-1)  # sentinel

    async def consumer():
        while True:
            item = await queue.get()
            if item == -1:
                break
            print(f"Consumed: {item}")
            await asyncio.sleep(0.2)

    await asyncio.gather(producer(), consumer())


async def main():
    urls = ["api.example.com/users", "api.example.com/posts", "api.example.com/comments"]
    results = await fetch_all_concurrent(urls)
    for r in results:
        print(f"  {r['url']} -> {r['status']}")

    print("\nProducer-consumer:")
    await producer_consumer()


if __name__ == "__main__":
    asyncio.run(main())
