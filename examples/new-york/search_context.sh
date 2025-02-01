#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.search_context \
        --query "a person eating pizza walking on the street" \
