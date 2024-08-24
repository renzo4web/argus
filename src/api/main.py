from typing import List
from fastapi import FastAPI, APIRouter, BackgroundTasks
from pydantic import BaseModel, field_validator, HttpUrl
import re
import json
import httpx
from argus.crew import ArgusCrew
from fastapi_versionizer.versionizer import Versionizer, api_version

app = FastAPI(title='Image Processing API', redoc_url=None)
image_router = APIRouter(prefix='/description', tags=['Image Processing'])

class ImageRequestBase(BaseModel):
    urls: List[str]

    @field_validator('urls')
    @classmethod
    def validate_image_urls(cls, urls):
        valid_extensions = r'\.(jpg|jpeg|png|gif|bmp|webp)$'
        for url in urls:
            if not re.search(valid_extensions, url.lower()):
                raise ValueError(f"Invalid image URL: {url}. URL must end with a valid image extension.")
        return urls

class ImageRequestV1(ImageRequestBase):
    webhook_url: HttpUrl

class ImageRequestV2(ImageRequestBase):
    pass

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

@api_version(1)
@image_router.post('')
def get_description(request: ImageRequestV1, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_images_and_send_webhook, request.urls, str(request.webhook_url))
    return {"status_code": 202, "message": "Processing started. Results will be sent to the provided webhook."}

@api_version(2)
@image_router.post('')
def get_description_v2(request: ImageRequestV2):
    try:
        crew = ArgusCrew().crew()
        inputs = {"image_paths_urls": request.urls}
        result = crew.kickoff(inputs=inputs)
        
        if result and result.raw:
            parsed_result = json.loads(result.raw)
            return {"status_code": 200, "data": parsed_result}
        else:
            return {"status_code": 422, "error": "Argus failed to process the images or returned no result"}
    except Exception as e:
        return {"status_code": 500, "error": f"An error occurred while processing the images: {str(e)}"}

app.include_router(image_router)

versions = Versionizer(
        app=app,
        prefix_format='/v{major}',
        semantic_version_format='{major}',
        latest_prefix='/latest',
        sort_routes=True
    ).versionize()
    