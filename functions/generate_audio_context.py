from openai import OpenAI
from pathlib import Path
import argparse
import json
import logging

# import ffmpeg
from moviepy.editor import VideoFileClip

client = OpenAI()


def process_video(
    video_file: Path, audio_file: Path, transcript_workspace: Path, temperature=0
):
    # Extract audio from video
    logging.info(f"Extracting audio from video {video_file}...")

    if not audio_file.exists():
        video = VideoFileClip(str(video_file))  # Load video
        video.audio.write_audiofile(str(audio_file))  # Save audio file

    # Extract transcript from audio
    # Transcription is saved in segments
    logging.info("Transcribing audio...")
    transcript_workspace.mkdir(parents=True, exist_ok=True)
    with open(audio_file, "rb") as audio:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # segments = []
    logging.info("Segmenting transcript...")
    for segment in transcription.segments:
        # print("raw segment:", segment)
        # input("Press Enter to continue...")

        start = round(segment.start)
        end = round(segment.end)
        raw_text = segment.text
        summary = client.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise description of this transcript, highlighting keywords for easy searchability:",
                },
                {"role": "user", "content": raw_text},
            ],
        )

        summary = summary.choices[0].message.content.strip()
        # print(
        #     f"after process, start: {start}, end: {end}, summary: {summary}, raw_text: {raw_text}"
        # )
        # input("Press Enter to continue...")

        # Save segment to a separate file
        segment_data = {
            "summary": summary,
            "raw_text": raw_text,
        }
        segment_filename = f"{transcript_workspace}/{start}-{end}.json"
        with open(segment_filename, "w", encoding="utf-8") as f:
            json.dump(segment_data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parsed = parser.parse_args()

    workspace: Path = parsed.workspace
    assert workspace.exists(), "Workspace does not exist."
    sourceFolder = Path(parsed.workspace, "source")
    assert sourceFolder.exists(), "Source folder does not exist."

    audioFolder = Path(workspace, "audio")
    audioFolder.mkdir(parents=True, exist_ok=True)
    transcriptsFolder = Path(workspace, "audio_transcripts")
    transcriptsFolder.mkdir(parents=True, exist_ok=True)

    for videoFile in sourceFolder.glob("*.MOV"):
        stem = videoFile.stem
        segments = process_video(
            video_file=videoFile,
            audio_file=Path(audioFolder, f"{stem}.mp3"),
            transcript_workspace=Path(transcriptsFolder, stem),
        )


if __name__ == "__main__":
    main()
