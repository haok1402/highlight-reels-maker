#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.generate_video_context \
        --workspace hackathon
