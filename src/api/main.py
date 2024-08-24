import json
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, field_validator, HttpUrl
from typing import List
import re
import httpx
from argus.crew import ArgusCrew

app = FastAPI()

class ImageRequest(BaseModel):
    urls: List[str]
    webhook_url: HttpUrl

    @field_validator('urls')
    @classmethod
    def validate_image_urls(cls, urls):
        valid_extensions = r'\.(jpg|jpeg|png|gif|bmp|webp)$'
        for url in urls:
            if not re.search(valid_extensions, url.lower()):
                raise ValueError(f"Invalid image URL: {url}. URL must end with a valid image extension.")
        return urls

def process_images_and_send_webhook(urls: List[str], webhook_url: str):
    try:
        crew = ArgusCrew().crew()
        print("Received image URLs: ", urls)
        print("Webhook URL: ", webhook_url)
        inputs = {"image_paths_urls": urls}
        result = crew.kickoff(inputs=inputs)
        
        if result and result.raw:
            parsed_result = json.loads(result.raw)
            status_code = 200
            response_data = parsed_result
        else:
            status_code = 422
            response_data = {"error": "Argus failed to process the images or returned no result"}
        
        with httpx.Client() as client:
            response = client.post(webhook_url, json={
                "status_code": status_code,
                "data": response_data
            })
            response.raise_for_status()
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        with httpx.Client() as client:
            client.post(webhook_url, json={
                "status_code": 500,
                "data": {"error": f"An error occurred while processing the images: {str(e)}"}
            })

@app.post("/description")
def get_description(request: ImageRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_images_and_send_webhook, request.urls, str(request.webhook_url))
    return {"message": "Processing started. Results will be sent to the provided webhook."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)