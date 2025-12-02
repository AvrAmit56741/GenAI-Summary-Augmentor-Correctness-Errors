
print("vLLM API Server v1 - For environment with transformers 4.57+")
# ============================================================================
# CONFIGURATION
# ============================================================================
# Binary Classifier
BINARY_MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"
BINARY_MAX_SEQ_LENGTH = 3072
BINARY_ENABLE_ATTENTION = True
BINARY_TOP_K_TOKENS = 15

# LLM Classifier (Gemma) - vLLM with merged model
LLM_MODEL_PATH = "/home/avramit/classifier/01122025/llmfinetune/hebrew_evidence_llm_lora_85p_merged/"
LLM_MAX_SEQ_LENGTH = 4096
LLM_GPU_MEMORY_UTILIZATION = 0.3  # Reduced to 30% (uses ~14GB instead of 24GB)
LLM_PROMPT_FORMAT = "alpaca"

PORT = 8081  # Different port to avoid conflict with unsloth server
HOST = "0.0.0.0"

# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODELS
# ============================================================================
binary_model = None
binary_tokenizer = None
binary_device = None
binary_config = None

llm_model = None
llm_tokenizer = None

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================
class BinaryClassifierRequest(BaseModel):
    premise: str
    hypothesis: str
    return_attention: bool = False

class AttentionToken(BaseModel):
    token: str
    weight: float

class BinaryClassifierResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    attention_top_tokens: Optional[List[AttentionToken]] = None

class LLMClassifierRequest(BaseModel):
    article: str
    summary: str
    claim: str
    prompt_format: Optional[str] = None

class LLMClassifierResponse(BaseModel):
    generated_text: str
    parsed_json: Optional[dict] = None

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="vLLM Multi-Model Inference API")

@app.on_event("startup")
async def load_models():
    """Load models once at server startup"""
    global binary_model, binary_tokenizer, binary_device, binary_config, llm_model, llm_tokenizer

    # Binary Classifier
    logger.info(f"Loading binary model from {BINARY_MODEL_PATH}...")
    try:
        model_kwargs = {"trust_remote_code": True}
        if BINARY_ENABLE_ATTENTION:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True

        binary_model = AutoModelForSequenceClassification.from_pretrained(
            BINARY_MODEL_PATH,
            **model_kwargs
        )
        binary_config = binary_model.config
        binary_tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH, trust_remote_code=True)

        binary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        binary_model.to(binary_device)
        binary_model.eval()

        logger.info(f"✅ Binary classifier loaded on {binary_device}")
        if hasattr(binary_config, 'id2label') and hasattr(binary_config, 'num_labels'):
            logger.info(f"Labels: {[binary_config.id2label[i] for i in range(binary_config.num_labels)]}")
    except Exception as e:
        logger.error(f"❌ Failed to load binary classifier: {e}")

    # LLM with vLLM
    logger.info(f"Loading LLM with vLLM from {LLM_MODEL_PATH}...")
    try:
        llm_model = LLM(
            model=LLM_MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            max_model_len=LLM_MAX_SEQ_LENGTH,
            trust_remote_code=True,
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
        logger.info(f"✅ LLM loaded with vLLM")
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}")
        logger.info("LLM endpoints will not be available")

@app.get("/")
async def root():
    return {
        "status": "vLLM Multi-Model Inference API",
        "port": PORT,
        "models": {
            "binary": {
                "path": BINARY_MODEL_PATH,
                "loaded": binary_model is not None
            },
            "llm": {
                "path": LLM_MODEL_PATH,
                "loaded": llm_model is not None,
                "mode": "vLLM"
            }
        },
        "binary_device": str(binary_device) if binary_device else None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "binary_model_loaded": binary_model is not None,
        "llm_model_loaded": llm_model is not None,
        "binary_device": str(binary_device) if binary_device else None
    }

@app.post("/binary/predict", response_model=BinaryClassifierResponse)
async def predict_binary(request: BinaryClassifierRequest):
    """Run binary NLI inference"""
    if binary_model is None or binary_tokenizer is None:
        raise HTTPException(status_code=503, detail="Binary model not loaded")

    try:
        inputs = binary_tokenizer(
            request.premise,
            request.hypothesis,
            return_tensors="pt",
            padding="max_length",
            max_length=BINARY_MAX_SEQ_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(binary_device) for k, v in inputs.items()}

        with torch.no_grad():
            if BINARY_ENABLE_ATTENTION and request.return_attention:
                outputs = binary_model(**inputs, output_attentions=True)
            else:
                outputs = binary_model(**inputs)

        logits = outputs.logits[0].cpu()
        probs = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class = binary_config.id2label[predicted_class_idx]
        confidence = probs[predicted_class_idx].item()

        probabilities = {
            binary_config.id2label[i]: float(probs[i].item())
            for i in range(binary_config.num_labels)
        }

        attention_top_tokens = None
        if BINARY_ENABLE_ATTENTION and request.return_attention and outputs.attentions is not None:
            attention_top_tokens = extract_attention(inputs, outputs.attentions)

        return BinaryClassifierResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            attention_top_tokens=attention_top_tokens
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/predict", response_model=LLMClassifierResponse)
async def predict_llm(request: LLMClassifierRequest):
    """Run LLM inference with vLLM"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")

    try:
        prompt_format = request.prompt_format if request.prompt_format else LLM_PROMPT_FORMAT

        if prompt_format == "alpaca":
            prompt = format_alpaca_prompt(request.article, request.summary, request.claim)
        elif prompt_format == "gemma":
            prompt = format_gemma_prompt(request.article, request.summary, request.claim)
        elif prompt_format == "chatml":
            prompt = format_chatml_prompt(request.article, request.summary, request.claim)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid prompt_format: {prompt_format}")

        sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=512)
        outputs = llm_model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        parsed_json = parse_json_output(generated_text)

        return LLMClassifierResponse(
            generated_text=generated_text,
            parsed_json=parsed_json
        )

    except Exception as e:
        logger.error(f"LLM prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_alpaca_prompt(article, summary, claim):
    instruction = "Analyze the context and identify any factual errors in the claim."
    input_text = f"CONTEXT: {article}\n\nSUMMARY: {summary}\n\nCLAIM: {claim}"
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def format_gemma_prompt(article, summary, claim):
    return f"""<bos><start_of_turn>user
You are a Hebrew factuality checker. Analyze the context and identify any factual errors in the claim.

CONTEXT: {article}

SUMMARY: {summary}

CLAIM: {claim}

Identify the error type and return as JSON.<end_of_turn>
<start_of_turn>model
"""

def format_chatml_prompt(article, summary, claim):
    return f"""<|im_start|>system
You are a Hebrew factuality checker specialized in identifying errors in summarized claims.<|im_end|>
<|im_start|>user
CONTEXT: {article}

SUMMARY: {summary}

CLAIM: {claim}

Identify the factual error type and return as JSON.<|im_end|>
<|im_start|>assistant
"""

def parse_json_output(output_text):
    import json
    import re
    try:
        return json.loads(output_text)
    except:
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None

def extract_attention(inputs, attentions) -> List[AttentionToken]:
    try:
        tokens = binary_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        last_layer_attn = attentions[-1]
        avg_attn = last_layer_attn.mean(dim=1)[0]
        cls_attention = avg_attn[0, :].cpu().numpy()

        content_start = 1
        content_end = len(tokens)
        if '[PAD]' in tokens:
            content_end = tokens.index('[PAD]')

        content_tokens = tokens[content_start:content_end]
        cls_attention_content = cls_attention[content_start:content_end]

        if cls_attention_content.sum() > 0:
            cls_attention_norm = cls_attention_content / cls_attention_content.sum()
        else:
            cls_attention_norm = cls_attention_content

        top_k = min(BINARY_TOP_K_TOKENS, len(content_tokens))
        top_indices = cls_attention_norm.argsort()[::-1][:top_k]

        return [
            AttentionToken(
                token=content_tokens[idx],
                weight=float(cls_attention_norm[idx])
            )
            for idx in top_indices
        ]
    except Exception as e:
        logger.error(f"Attention extraction error: {e}")
        return None

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

