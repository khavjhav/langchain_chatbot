from starlette.responses import StreamingResponse
import asyncio
from fastapi import FastAPI
from assistant import generator,embedJson,getContext,embedJson_without_thread,chat_generator
from pydantic import BaseModel
import uvicorn

app=FastAPI()

class Item(BaseModel):
    question: str
class ChatItem(BaseModel):
    messages: list


@app.post("/generate")
async def hello_world(item: Item):

    # return StreamingResponse(generator(question), media_type="text/event-stream")
    return generator(item.question)

@app.post("/chat")
async def hello_world(item: ChatItem):

    # return StreamingResponse(generator(question), media_type="text/event-stream")
    return chat_generator(item.messages)

@app.get("/embed")
async def hello_world():
    return embedJson()

@app.get("/embedWithoutThread")
async def hello_world():
    return embedJson_without_thread()
    # return {"message": "Hello"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
