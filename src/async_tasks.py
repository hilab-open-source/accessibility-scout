import asyncio

import src.constants as const


def async_rate_limit_parallelize(
    func_list: list,
    delay: float = const.DELAY_SECONDS,
    semaphore_max: int = const.SEMAPHORE_MAX,
    timeout: float = const.TIMEOUT,
):
    return asyncio.run(
        _async_rate_limit_parallelize(func_list, delay, semaphore_max, timeout)
    )


async def _async_rate_limit_parallelize(
    func_list: list,
    delay: float = const.DELAY_SECONDS,
    semaphore_max: int = const.SEMAPHORE_MAX,
    timeout: float = const.TIMEOUT,
):
    sem = asyncio.Semaphore(semaphore_max)

    async def _rate_limit_task(func, delay: float, sem: asyncio.Semaphore):
        async with sem:
            await asyncio.sleep(delay)
            return await func

    tasks = []
    for func in func_list:
        tasks.append(asyncio.create_task(_rate_limit_task(func, delay, sem)))

    done, pending = await asyncio.wait(tasks, timeout=timeout)
    print(f"Completed Tasks: {len(done)}/{len(done) + len(pending)}")

    # cancel any unfinished
    for fut in pending:
        fut.cancel()

    vals = [v.result() for v in done]

    return vals
