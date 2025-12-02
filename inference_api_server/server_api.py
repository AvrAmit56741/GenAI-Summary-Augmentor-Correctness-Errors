
print("v9 - Binary classifier only (LLM disabled to avoid conflicts)")
# ============================================================================
# ENVIRONMENT SEPARATION STRATEGY
# ============================================================================
# This server (bertclassifier env): Binary classifier with transformers 4.49.0
# Use server_api_vllm.py (vllm_env): Both models with transformers 4.57+
#
# Reason: NeoBERT custom model requires transformers 4.49.0, but unsloth/vLLM
# require transformers 4.57+. They cannot coexist in the same environment.
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================
# Binary Classifier (using ORIGINAL working variable names)
BINARY_MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"
BINARY_MAX_SEQ_LENGTH = 3072
BINARY_ENABLE_ATTENTION = True  # Set to False for faster inference
BINARY_TOP_K_TOKENS = 15

# LLM Classifier (Gemma)
# NOTE: LLM (unsloth) requires newer transformers that conflicts with binary classifier
# Set ENABLE_LLM = False to only serve binary classifier
ENABLE_LLM = False  # Set to False to avoid unsloth/transformers conflicts
LLM_MODEL_PATH = "/home/avramit/classifier/01122025/llmfinetune/hebrew_evidence_llm_lora_85p/"
LLM_MAX_SEQ_LENGTH = 4096
LLM_USE_VLLM = False
LLM_PROMPT_FORMAT = "alpaca"  # Options: "alpaca", "gemma", "chatml"

PORT = 8080
HOST = "0.0.0.0"

# ============================================================================
# IMPORTS (Keep transformers clean, no unsloth imports here!)
# ============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODELS
# ============================================================================
# Binary classifier (using ORIGINAL working variable names)
binary_model = None
binary_tokenizer = None
binary_device = None
binary_config = None

# LLM classifier
llm_model = None
llm_tokenizer = None
llm_device = None

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
app = FastAPI(title="Multi-Model Inference API")

@app.on_event("startup")
async def load_models():
    """Load models once at server startup"""
    global binary_model, binary_tokenizer, binary_device, binary_config
    global llm_model, llm_tokenizer, llm_device

    # ========================================================================
    # BINARY CLASSIFIER
    # ========================================================================
    logger.info(f"Loading binary model from {BINARY_MODEL_PATH}...")
    try:
        # Load model first
        model_kwargs = {"trust_remote_code": True}
        if BINARY_ENABLE_ATTENTION:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True

        binary_model = AutoModelForSequenceClassification.from_pretrained(
            BINARY_MODEL_PATH,
            **model_kwargs
        )
        
        # Get config from loaded model
        binary_config = binary_model.config
        
        # Load tokenizer
        binary_tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH, trust_remote_code=True)

        # Move to device
        binary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        binary_model.to(binary_device)
        binary_model.eval()

        logger.info(f"✅ Binary classifier loaded on {binary_device}")
        if hasattr(binary_config, 'id2label') and hasattr(binary_config, 'num_labels'):
            logger.info(f"Labels: {[binary_config.id2label[i] for i in range(binary_config.num_labels)]}")
    except Exception as e:
        logger.error(f"❌ Failed to load binary classifier: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # LLM CLASSIFIER - Load separately with unsloth
    # ========================================================================
    if not ENABLE_LLM:
        logger.info("LLM loading disabled (ENABLE_LLM=False). Only binary classifier available.")
        return
    
    logger.info(f"Loading LLM from {LLM_MODEL_PATH}...")
    try:
        if LLM_USE_VLLM:
            try:
                from vllm import LLM
            except ImportError:
                logger.error("❌ vLLM not installed. Install with: pip install vllm")
                raise
            
            llm_model = LLM(
                model=LLM_MODEL_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5,
                max_model_len=LLM_MAX_SEQ_LENGTH,
                trust_remote_code=True,
            )
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
            llm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"✅ LLM loaded with vLLM on {llm_device}")
        else:
            try:
                from unsloth import FastLanguageModel
            except ImportError:
                logger.error("❌ Unsloth not installed. Install with: pip install unsloth")
                raise
            
            llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(
                model_name=LLM_MODEL_PATH,
                max_seq_length=LLM_MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(llm_model)
            llm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"✅ LLM loaded with unsloth (4-bit) on {llm_device}")
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}")
        logger.info("LLM endpoints will not be available")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {
        "status": "Multi-Model Inference API",
        "models": {
            "binary": {
                "path": BINARY_MODEL_PATH,
                "loaded": binary_model is not None
            },
            "llm": {
                "path": LLM_MODEL_PATH,
                "loaded": llm_model is not None,
                "mode": "vLLM" if LLM_USE_VLLM else "unsloth"
            }
        },
        "binary_device": str(binary_device) if binary_device else None,
        "llm_device": str(llm_device) if llm_device else None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "binary_model_loaded": binary_model is not None,
        "llm_model_loaded": llm_model is not None,
        "binary_device": str(binary_device) if binary_device else None,
        "llm_device": str(llm_device) if llm_device else None
    }

# ============================================================================
# BINARY CLASSIFIER ENDPOINT - EXACT WORKING CODE
# ============================================================================
@app.post("/binary/predict", response_model=BinaryClassifierResponse)
async def predict_binary(request: BinaryClassifierRequest):
    """Run binary NLI inference on premise-hypothesis pair"""

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

# ============================================================================
# LLM CLASSIFIER ENDPOINT
# ============================================================================
@app.post("/llm/predict", response_model=LLMClassifierResponse)
async def predict_llm(request: LLMClassifierRequest):
    """Run LLM inference for error detection"""

    if llm_model is None or llm_tokenizer is None:
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

        if LLM_USE_VLLM:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=512)
            outputs = llm_model.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
        else:
            inputs = llm_tokenizer(prompt, return_tensors="pt")
            if llm_device:
                inputs = {k: v.to(llm_device) for k, v in inputs.items()}
            
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id
            )
            full_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_text.split("### Response:")[-1].strip() if "### Response:" in full_text else full_text

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
    """Format prompt in Alpaca style"""
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
    """Format prompt in Gemma native style"""
    return f"""<bos><start_of_turn>user
You are a Hebrew factuality checker. Analyze the context and identify any factual errors in the claim.

CONTEXT: {article}

SUMMARY: {summary}

CLAIM: {claim}

Identify the error type and return as JSON.<end_of_turn>
<start_of_turn>model
"""


def format_chatml_prompt(article, summary, claim):
    """Format prompt in ChatML style (OpenAI standard)"""
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
    """Try to parse JSON from model output"""
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
    """Extract top K tokens by attention weight"""
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
