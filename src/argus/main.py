#!/usr/bin/env python
from datetime import datetime
import sys
from argus.crew import ArgusCrew

def run(image_urls):
    """
    Run the crew with the given image URLs.
    """
    inputs = {
        'image_paths_urls': image_urls
    }
    crew_output = ArgusCrew().crew().kickoff(inputs=inputs)
    # save the raw output to a json file
    with open(f'raw_output-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w') as file:
        file.write(crew_output.raw)
    return crew_output.raw

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        ArgusCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ArgusCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        ArgusCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")