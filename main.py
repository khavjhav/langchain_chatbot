from starlette.responses import StreamingResponse
import asyncio
from fastapi import FastAPI
from assistant import generator,embedJson,getContext
from pydantic import BaseModel


app=FastAPI()

class Item(BaseModel):
    question: str


@app.post("/generate")
async def hello_world(item: Item):

    # return StreamingResponse(generator(question), media_type="text/event-stream")
    return generator(item.question)

@app.get("/embed")
async def hello_world():
    return embedJson()
    # return {"message": "Hello"}
