"""
Generate description for key frames in the video. Since the use case is
primarily for vloggers, we assume each scene would be at least a few seconds
apart. As a result, we sample one frame per second from the video and generate
description for that frame. The description is generated using OpenAI's gpt-4o
model. Instead of generating a text-aligned embedding from the frame, we choose
to describe the frame directly, so multiple objects in the frame can be better
discerned during the lookup process.

@author: Hao Kang <haok1402@gmail.com>
@date: February 1, 2025
"""

import base64
import asyncio
import logging
import argparse
import subprocess
from typing import List
from pathlib import Path

import openai
import aiofiles


def extract_frames(workspace: Path, videoFile: Path) -> List[Path]:
    """
    Extract frames 1-second apart from the video file and save locally.
    Each frame is downsampled to 1/4 of the original resolution to save space.

    :param workspace: Path to the workspace.
    :param videoFile: Path to the video file.
    :return: List of paths to the extracted frames.
    """
    if workspace.exists():
        return list(workspace.glob("*.jpg"))

    logging.info(f"Extracting frames from {videoFile}.")
    workspace.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            videoFile,
            "-vf",
            "fps=1,scale=iw/4:ih/4",
            f"{workspace}/frame_%04d.jpg",
        ],
    )

    return list(workspace.glob("*.jpg"))


async def describe_frame(
    workspace: Path,
    client: openai.AsyncClient,
    frameFile: Path,
    semaphore: asyncio.Semaphore,
):
    """
    Generate description for a single frame using OpenAI's GPT-4 model.

    :param workspace: Path to the workspace.
    :param client: OpenAI client.
    :param frameFile: Path to the frame file.
    :param semaphore: Semaphore to limit the number of concurrent requests.
    """
    logging.info(f"Describing {frameFile}.")

    async with aiofiles.open(frameFile, "rb") as f:
        frame = await f.read()
        base64_frame = base64.b64encode(frame).decode("utf-8")
    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise description of the image, highlighting key objects and features for easy searchability.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frame}"
                            },
                        },
                    ],
                },
            ],
        )
    async with aiofiles.open(Path(workspace, frameFile.stem + ".txt"), "w") as f:
        await f.write(response.choices[0].message.content)


async def describe(workspace: Path, keyframes: List[Path]):
    """
    Generate description for each key frame using OpenAI's GPT-4 model.

    :param workspace: Path to the workspace.
    :param keyframes: List of key frames.
    """

    if workspace.exists():
        return

    logging.info(f"Describing key frames in {workspace}.")
    workspace.mkdir(parents=True, exist_ok=True)

    client = openai.AsyncClient()
    semaphore = asyncio.Semaphore(2)  # stay under the rate limit

    await asyncio.gather(
        *[describe_frame(workspace, client, frame, semaphore) for frame in keyframes]
    )


async def main():
    """
    Generate description for key frames in the video.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parsed = parser.parse_args()

    workspace: Path = parsed.workspace
    assert workspace.exists(), "Workspace does not exist."
    sourceFolder = Path(parsed.workspace, "source")
    assert sourceFolder.exists(), "Source folder does not exist."

    keyframesFolder = Path(workspace, "video_keyframes")
    keyframesFolder.mkdir(parents=True, exist_ok=True)
    contextFolder = Path(workspace, "video_context")
    contextFolder.mkdir(parents=True, exist_ok=True)

    for videoFile in sourceFolder.glob("*.MOV"):
        stem = videoFile.stem
        frames = extract_frames(Path(keyframesFolder, stem), videoFile)
        await describe(Path(contextFolder, stem), frames)


if __name__ == "__main__":
    asyncio.run(main())
