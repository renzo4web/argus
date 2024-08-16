from datetime import datetime
import sys
import json
import warnings
from PIL import Image
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv
import google.generativeai as genai
from textwrap import dedent
import os
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

# Suppress UserWarnings temporarily
warnings.filterwarnings("ignore", category=UserWarning)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.2, convert_system_message_to_human=True
)


class Vision:
    @tool("Product image classifier")
    def vision(inp):
        """
        This is a tool used to classify images

        Parameters:
        - inp: A JSON string containing 'prompt' and 'path' keys

        Returns:
        Response from the vision model based on the prompt and image provided
        """
        try:
            inp_dict = json.loads(inp)
            prompt = inp_dict.get("prompt")
            path = inp_dict.get("path")

            if not prompt or not path:
                return "Error: Both 'prompt' and 'path' are required in the input JSON."

            print(f"Image path: {path}")
            image = Image.open(path)
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            response = model.generate_content([prompt, image])

            return response.text
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide a valid JSON string."
        except Exception as e:
            return f"Error: {str(e)}"


vision_tool = Vision.vision

class ProductInfo(BaseModel):
    description: str
    tags: list[str]
    title: str
    type: str
    category: str

def json_summary_tool(inp):
    """
    This tool summarizes the extracted information into a structured JSON object.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a JSON summarizer. Extract the product information and format it into a structured JSON object."},
            {"role": "user", "content": str(inp)},
        ],
        response_format=ProductInfo,
    )

    response = completion.choices[0].message.parsed
    return response


# Agent responsible for loading images
image_analyst = Agent(
    role="Image Analyst",
    goal="Analyse the contents of the image and give all the information that you can about the product",
    backstory=dedent("""
        You are one of the most experienced product image classifier to ever exist in this world.
        you have knowledge of all images every clicked, you are also a product description expert.
        You dont miss any details of the image and do not invent false information.
    """),
    tools=[vision_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Agent that take the product information and create a description, tags, title, type, and category
product_description_agent = Agent(
    role="Product Description Agent",
    goal="Provide with all the information about the product in a structured way.",
    backstory=dedent("""
        You are a expert product categorizer, analyzer, and the best sales person in the world.
        You are also a expert product description agent.

        If you are able to provide with all the information about the product in a structured way, you will get a $10,000 commission!
        If you not have sufficient information only provide with the information that you have, do not provide with information that you do not have.
    """),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Agent responsible for summarizing the extracted data into a JSON object
json_summary_agent = Agent(
    role="JSON Summarizer",
    goal="Compile extracted information into a structured JSON object.",
    verbose=True,
    backstory="""
    You are a JSON Summarizer agent that specializes in compiling extracted information into a structured JSON object.
    You use a special tool to ensure the output is always in the correct format.
    """,
    # tools=[json_summary_tool],
    # llm=llm,
)

# Task Definitions
# ----------------
# Tasks define the actions that each agent will perform. These are dynamically generated based on the input file.


def get_tasks_for_file(file_path):
    """Generate a list of tasks to process a given image file."""
    tasks = [
        Task(
            description=dedent(f"""
                Image path: {file_path}
            Identify if is a product, what type of product is and give all the information about the product.
        If you do your BEST WORK, I'll give you a $10,000 commission!
      """),
            agent=image_analyst,
            expected_output="the product name and type of product and  brief description of the product.",
            verbose=True,
            max_iter=1
        ),
        Task(
            description="Create a description, tags, title, type, and category for the product.",
            agent=product_description_agent,
            expected_output=dedent( """
                 Create the following fields:
                - ProductName
                - Description
                - SellerPitch
                - Tags
                - Category

                the field in english and the value in spanish.
            """),
            verbose=True,
            max_iter=1
        )
    ]
    return tasks


# Main Function
# -------------
# This function orchestrates the entire process, from loading the image file to saving the processed data.


def main(file_path):
    """Main function to process the receipt image and save the extracted data as a JSON file."""
    # Check if the file exists
    # Define the crew and process
    crew = Crew(
        agents=[image_analyst, product_description_agent, json_summary_agent],
        tasks=get_tasks_for_file(file_path),
        process=Process.sequential,  # Define the process flow as sequential
        verbose=True,
    )

    # Run the crew to process the image and extract data
    result = crew.kickoff()

    json_output = json_summary_tool(result.raw)

    json = json_output.model_dump_json()
    # save the pydantic object to a json file
    with open(f"product_info_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
        f.write(json)
    print("Result: ", json)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_receipts.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)