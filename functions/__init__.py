import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logging/{int(time.time())}.log"),
        logging.StreamHandler(),
    ],
)
