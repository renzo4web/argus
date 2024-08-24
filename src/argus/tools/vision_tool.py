import base64
from textwrap import dedent
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
                    {"type": "text", "text": dedent("""
                        Analiza detalladamente estas imágenes de producto y proporciona la siguiente información:

                            1. Tipo de producto: ¿Qué es exactamente?
                            2. Descripción: ¿Puedes describir el producto en detalle? NO INVENTES NADA.
                            3. Características físicas: Color, material, tamaño (si es evidente), forma.
                            4. Marca: ¿Hay algún logo o nombre de marca visible?
                            5. Funcionalidad: ¿Cuál parece ser el uso principal del producto?
                            6. Calidad: ¿Qué puedes decir sobre la calidad aparente del producto?
                            7. Detalles únicos: ¿Hay alguna característica especial o distintiva?
                            8. Estado: ¿El producto parece nuevo, usado, vintage?
                            9. Embalaje: ¿Cómo viene presentado el producto (si es visible)?
                            10. Accesorios: ¿Viene con algún accesorio o componente adicional?
                            11. Contexto: ¿Hay algo en la imagen que sugiera el entorno de uso del producto?

                        Por favor, estructura tu respuesta en secciones claras. Si no puedes determinar algún aspecto con certeza, indica 'No se puede determinar' para ese punto específico.
                        """)}
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