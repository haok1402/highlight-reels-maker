#!/usr/bin/sh

source .env

conda run -n maker \
    python3 -m functions.search_context \
        --query "a person walking on the street holding a pizza"
