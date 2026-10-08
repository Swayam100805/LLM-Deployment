from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="LLM Deployment API")

# Load pretrained model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Request format
class PromptRequest(BaseModel):
    prompt: str


# Home endpoint
@app.get("/")
def home():
    return {"message": "LLM API is running"}


# LLM inference endpoint
@app.post("/generate")
def generate(request: PromptRequest):

    inputs = tokenizer(
        request.prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=10,
        do_sample=False,
        num_beams=4
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "response": response
    }