"""
Find the relevant frames in the video for the given query. The query is used to search for the frames that match the objects / features mentioned. The query is embedded with OpenAI's text-embedding-3-small model, and the frames are retrieved using Pinecone's vector similarity search service.

@author: Hao Kang <haok1402@gmail.com>
@date: February 1, 2025
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, DefaultDict, Tuple

import openai
import datetime
from pinecone.grpc import PineconeGRPC as Pinecone


def findHeavyRange(
    videoMatches: List[Dict[str, Any]], interval: int = 4
) -> Tuple[str, int, int]:
    """
    Find the range of frames that contain most retrieved frames.
    """

    frames: DefaultDict[str, List[str]] = defaultdict(list)
    for item in videoMatches:
        itemID = item["id"]
        videoName = str(Path(itemID).parent).replace("video_context", "source")
        frames[videoName].append(int(Path(itemID).stem.replace("frame_", "")))

    videoName, videoCount = "", 0
    for key, val in frames.items():
        if len(val) > videoCount:
            videoName, videoCount = key, val

    buffer = [0 for _ in range(max(frames[videoName]) + 1)]
    for index in frames[videoName]:
        buffer[index] += 1  # ffmpeg starts from 1 with frame index

    startIndex, count = 0, 0
    for i in range(1, len(buffer) - interval):
        csm = sum(buffer[i : i + interval])
        if csm > count:
            startIndex, count = i, csm

    videoName += ".MOV"
    return videoName, startIndex, startIndex + interval


def main():
    """
    Search for the relevant frames in the video for the given query.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    # parser.add_argument("--dumpPath", type=str, required=True)
    parsed = parser.parse_args()

    client = openai.Client()
    response = client.embeddings.create(
        input=parsed.query,
        model="text-embedding-3-small",
    )
    embedding = response.data[0].embedding

    pineconeApiKey = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pineconeApiKey)

    videoIndex = pc.Index("video-context")
    videoResults = videoIndex.query(
        vector=embedding,
        top_k=3,
        include_values=False,
        include_metadata=True,
    )

    print(videoResults.matches)

    # Use video as the primary search context and let audio be the secondary
    # heuristics for finding the relevant frames
    # vName, sIndex, eIndex = findHeavyRange(videoResults.matches)
    # sTime = str(datetime.timedelta(seconds=sIndex))
    # eTime = str(datetime.timedelta(seconds=eIndex))

    # # Extract the clips from the video.
    # subprocess.run(
    #     [
    #         "ffmpeg",
    #         "-y",
    #         "-i",
    #         vName,
    #         "-ss",
    #         sTime,
    #         "-to",
    #         eTime,
    #         "-c",
    #         "copy",
    #         parsed.dumpPath,
    #     ],
    # )


if __name__ == "__main__":
    main()
