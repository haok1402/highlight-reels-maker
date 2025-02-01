#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.generate_audio_context \
        --workspace new-york
