import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from tests.temporalio.activities import (
    fetch_file,
    load_file,
    split_text,
    index_into_milvus,
    index_into_es,
    summary,
)

async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="my-task-queue",
        workflows=[],
        activities=[
            fetch_file,
            load_file,
            split_text,
            index_into_milvus,
            index_into_es,
            summary,
        ],
        debug_mode=True,
        max_concurrent_workflow_tasks=8,

    )
    print("Worker started with DependentActivitiesWorkflow and all activities.")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
