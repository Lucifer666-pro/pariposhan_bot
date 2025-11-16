import os
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from rag_engine import RAGEngine
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()
rag = RAGEngine()

@app.get("/")
def home():
    return {"message": "Pariposhan bot is running!"}

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.form()
    msg = data.get("Body", "")
    
    print("Incoming message:", msg)

    # RAG answer
    answer = rag.query(msg)

    # Twilio response
    resp = MessagingResponse()
    resp.message(answer)

    return PlainTextResponse(str(resp), media_type="application/xml")
