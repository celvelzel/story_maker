#!/usr/bin/env python3
"""Local inference server for NLG models using Hugging Face transformers.

This script creates a simple FastAPI server that provides an OpenAI-compatible
API endpoint for local model inference. It loads a model from the specified
directory and serves it on the configured port.

Usage:
    python scripts/local_inference_server.py --model_path models/nlg/merged_model_Qwen3-4B-Instruct-2507/ --port 8000
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class InferenceServer:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        logger.info(f"Initializing inference server with model: {self.model_path}")
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        logger.info("Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

        if self.device == "auto":
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                logger.info("Using CUDA with automatic device mapping")
            else:
                logger.info("CUDA not available, using CPU")
        else:
            model_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **model_kwargs
        )

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        formatted_prompt = self._format_messages(messages)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif hasattr(self.model, 'hf_device_map'):
            first_device = next(iter(self.model.hf_device_map.values()))
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}\n\n")
            elif role == "user":
                formatted_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}\n\n")

        formatted_parts.append("Assistant: ")
        return "".join(formatted_parts)


app = FastAPI(title="Local NLG Inference Server", version="1.0.0")
inference_server: Optional[InferenceServer] = None





@app.get("/")
async def root():
    return {"message": "Local NLG Inference Server is running"}


@app.get("/health")
async def health_check():
    if inference_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if inference_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        start_time = time.time()
        response_text = inference_server.generate_response(
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1024
        )
        elapsed_time = time.time() - start_time

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

        logger.info(f"Generated response in {elapsed_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Local NLG Inference Server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/nlg/merged_model_Qwen3-4B-Instruct-2507/",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )

    args = parser.parse_args()

    global inference_server
    try:
        inference_server = InferenceServer(
            model_path=args.model_path,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference server: {e}")
        sys.exit(1)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()