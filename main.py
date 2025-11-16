from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from rag_engine import answer_from_rag
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

@app.post("/webhook")
async def twilio_webhook(
    Body: str = Form(...),
    From: str = Form(...)
):
    user_msg = Body
    phone = From  # User's WhatsApp number

    reply_text = answer_from_rag(user_msg)

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(reply_text)

    return PlainTextResponse(str(resp))

@app.get("/")
def home():
    return {"status": "Twilio WhatsApp bot running!"}
