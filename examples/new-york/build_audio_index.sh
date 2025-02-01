#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.build_audio_index \
        --workspace new-york
