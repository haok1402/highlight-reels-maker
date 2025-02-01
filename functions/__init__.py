import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("main.log", mode="w"), logging.StreamHandler()],
    format="%(asctime)s | %(levelname)s | %(message)s",
)
