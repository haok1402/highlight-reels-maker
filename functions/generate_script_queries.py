from openai import OpenAI
import json
import argparse
from pathlib import Path
import os
import re


def gen_script_queries(script_path, query_path, temperature=0.2):
    """
    Generate a list of queries from a video script using the GPT-4 model.
    """
    client = OpenAI()
    with open(script_path, "r") as f:
        script = f.read()

    # Call the ChatCompletion endpoint
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that generates concise, natural language queries "
                    'for AI-powered video retrieval. Provide your final output in an array of query strings braced by "[]", no additional keys.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Given the following script for a video, generate a list of concise, natural language queries that can be used for AI-powered video retrieval. Each query should directly describe a moment in the video. The queries should be structured as if they are directly describing what is happening in the footage. Script: {script}"
                ),
            },
        ],
    )

    # Extract the model's answer
    match = re.search(r"\[(.*)\]", response.choices[0].message.content)
    queries = json.loads("[" + match.group(1) + "]") if match else []

    # print("response:", response)
    # print(f"Queries : {queries}")
    # input("Press Enter to continue...")

    # Save the queries to a JSON file
    with open(query_path, "w", encoding="utf-8") as f:
        json.dump(queries, f)

    print(f"Queries have been saved to {query_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=Path)
    arg = parser.parse_args()

    script_path: Path = arg.script_path
    query_path = str(script_path).split(".")[0] + "_queries.json"
    assert os.path.exists(script_path), "script_path does not exist."

    gen_script_queries(script_path, query_path)


if __name__ == "__main__":
    main()
