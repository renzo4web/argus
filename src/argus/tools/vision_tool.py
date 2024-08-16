import base64
from typing import Type, List

from crewai_tools.tools.base_tool import BaseTool
from openai import OpenAI
from pydantic.v1 import BaseModel


class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_paths_urls: List[str] = []


class VisionTool(BaseTool):
    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of one or more images."
    )
    args_schema: Type[BaseModel] = ImagePromptSchema

    def _run(self, **kwargs) -> str:
        client = OpenAI()

        image_paths_urls = kwargs.get("image_paths_urls", [])

        if not image_paths_urls:
            return "Image Paths or URLs are required."

        print("VisionTool image_paths_urls: ", image_paths_urls)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What type of product is in these images? Is there any difference between them? please provide a detailed description of the product, DO NOT miss any important information"},
                ]
            }
        ]

        for url in image_paths_urls:
            if url.startswith('http'):
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
            else:
                base64_image = self._encode_image(url)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
        )

        return response.choices[0].message.content

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")