#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.generate_script_queries \
        --script_path script.txt
