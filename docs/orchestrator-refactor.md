  ## Orchestrator Refactor Plan

  ### Goals
  - Keep all orchestration logic on a single asyncio event loop.
  - Offload blocking helper work via asyncio primitives rather than nested loops.
  - Bound long-running operations (browser calls, downloads) with explicit timeouts.
  - Ensure helper tasks always signal completion or failure back to the main loop.

  ### Action Items
  1. Expose an async helper runtime entry point (e.g. `python_runtime.execute_async`).
  2. Schedule helpers using `asyncio.create_task` / `asyncio.TaskGroup`; avoid `run_coroutine_threadsafe`.
  3. Wrap Playwright or other IO-bound calls in `asyncio.wait_for` with configurable per-URL and overall timeouts.
  4. Scope browser instances per event loop and perform deterministic cleanup after each helper.
  5. Return helper results via awaited tasks so the orchestrator can emit `helpers_execution_end` without polling.
  6. Keep the overall `download()` call guarded by a top-level timeout to avoid hanging helpers.

  ---

  ### Implementation Notes

  To keep SABRE’s orchestration loop resilient (and avoid the worker-thread/loop traps we saw), aim for single event-loop ownership and push blocking work into
  bounded async tasks:

  1. **Keep the orchestrator fully async**
     - No `asyncio.run_coroutine_threadsafe` or ad-hoc loops inside helpers.
     - Schedule helper work from the orchestrator’s loop via `asyncio.create_task` or wrap blocking blocks in `asyncio.to_thread`, then `await`.

  2. **Expose an async helper runtime API**
     - Replace `run_async_from_sync` calls with something like `python_runtime.execute_async`.
     - If synchronous work (e.g., matplotlib) is unavoidable, isolate it inside `asyncio.to_thread` so control returns to the loop immediately.

  3. **Bound long-lived operations**
     - Wrap Playwright and other slow IO in `asyncio.wait_for` so hung helpers exit cleanly and can emit errors.
     - Offer per-URL timeouts (like the download helper) plus an overall helper timeout to keep responses snappy.

  4. **Clean up background tasks deterministically**
     - After each helper, close pages/contexts (Playwright, etc.) from the same event loop.
     - Avoid shared class-level state unless it’s explicitly keyed per loop.

  5. **Return helper completion via awaited tasks**
     - Have `_execute_helpers` yield `asyncio.Task` objects (or `await` them inline) so completion is natural—no polling or thread exceptions bubbling up.

  6. **Use structured concurrency**
     - In Python 3.11+, `asyncio.TaskGroup` is a clean way to manage helper lifetimes:
       ```python
       async with asyncio.TaskGroup() as tg:
           for helper in helpers:
               tg.create_task(run_helper(helper))
       ```
       Everything finishes or cancels together with no dangling tasks.

  7. **Guard external services behind adapters**
     - Keep Playwright/Selenium behind an async adaptor returning plain Python types.
     - Instantiate per event loop (e.g., via context managers) so each helper gets a clean instance.

  With that layout the orchestrator only awaits coroutines scheduled on its loop. Helpers can still run blocking code via `to_thread`, but they hand results back
  through awaitables—eliminating the re-entrancy deadlocks we hit before.
  EOF

