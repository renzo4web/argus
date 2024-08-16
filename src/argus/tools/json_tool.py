import os
from typing import Type, Any
from pydantic import BaseModel
from crewai_tools.tools.base_tool import BaseTool
from openai import OpenAI

class ProductInfo(BaseModel):
    description: str
    tags: list[str]
    title: str
    type: str
    category: str

class JsonToolSchema(BaseModel):
    """Input for JSON Tool."""
    input_text: Any

class JsonTool(BaseTool):
    name: str = "JSON Tool"
    description: str = "This tool summarizes the extracted information into a structured JSON object."
    args_schema: Type[BaseModel] = JsonToolSchema

    def _run(self, **kwargs) -> str:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        print("json_tool input: ", kwargs)
        input_text = kwargs.get("input_text", "")
        
        print("input text", input_text)
        if not input_text:
            return "Input text is required."

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON summarizer. Extract the product information and format it into a structured JSON object."},
                {"role": "user", "content": str(input_text)},
            ],
            response_format=ProductInfo,
        )

        response = completion.choices[0].message.parsed
        return  response.model_dump_json()
