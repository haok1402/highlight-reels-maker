"""
Build the dense index for retrieval of audio transcript segments. The index is built using
Pinecone's vector similarity search service, which allows for out-of-the-box
dense indexing and metadata storage. The index is built using the OpenAI's
text-embedding-3-small model, which generates a 1536-dimensional embedding
for each transcript segment summary, which allows for interactive search.

@author: Hao Kang <haok1402@gmail.com>
@date: February 1, 2025
"""

import os
import time
import asyncio
import argparse
import logging
from pathlib import Path
import json

import openai
import aiofiles
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


async def store_frame(
    client: openai.AsyncClient,
    semaphore: asyncio.Semaphore,
    contextFile: Path,
    index: Pinecone.Index,
):
    """
    Store the transcript segment summary embedding in the Pinecone index.

    :param client: OpenAI client
    :param semaphore: Semaphore for rate limiting
    :param contextFile: Path to the context file
    :param index: Pinecone index
    """
    logging.info(f"Storing {contextFile} in the index.")

    async with aiofiles.open(contextFile, "r", encoding="utf-8") as f:
        content = await f.read()

    data = json.loads(content)
    summary = data.get("summary", "")
    print(f"Summary of {str(contextFile)}:", summary)
    input("Press Enter to continue...")

    async with semaphore:
        response = await client.embeddings.create(
            input=summary,
            model="text-embedding-3-small",
        )
    embedding = response.data[0].embedding
    index.upsert(
        vectors=[
            {
                "id": str(contextFile),
                "values": embedding,
                "metadata": {
                    "summary": summary,
                },
            }
        ]
    )


async def main():
    """
    Build the dense index for retrieval of audio frames.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parsed = parser.parse_args()

    workspace: Path = parsed.workspace
    assert workspace.exists(), "Workspace does not exist."

    client = openai.AsyncClient()
    semaphore = asyncio.Semaphore(4)  # Stay under the rate limit

    pineconeApiKey = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pineconeApiKey)
    if not pc.has_index("audio-context"):
        pc.create_index(
            "audio-context",
            metric="cosine",
            dimension=1536,  # text-embedding-3-small
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index("audio-context")
    contextFolder = Path(workspace, "audio_transcripts")
    await asyncio.gather(
        *[
            store_frame(client, semaphore, contextFile, index)
            for contextFile in contextFolder.rglob("*.txt")
        ]
    )

    # I don't particularly like this approach, in that we're polling the index
    # status constantly. However, it's provided in the official documentation.
    while not pc.describe_index("audio-context").status["ready"]:
        time.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
