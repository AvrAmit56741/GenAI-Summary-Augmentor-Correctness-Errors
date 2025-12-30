
print("v17 - Added binary v2 and multilabel setup2 v2 model support")
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

print("v1")
# Binary Classifier v1 (using ORIGINAL working variable names)
#BINARY_V1_MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"
BINARY_V1_MODEL_PATH = "/home/avramit/classifier/17122025/classifier/binary/neodictabert-finetuned-binary-171225/"
BINARY_V1_MAX_SEQ_LENGTH = 3072
BINARY_V1_ENABLE_ATTENTION = True  # Set to False for faster inference
BINARY_V1_TOP_K_TOKENS = 15

# Binary Classifier v2 (new model)
BINARY_V2_MODEL_PATH = "/home/avramit/classifier/04122025/classifier/binary/neodictabert-finetuned-binary-041225/"
BINARY_V2_MAX_SEQ_LENGTH = 3072
BINARY_V2_ENABLE_ATTENTION = True  # Set to False for faster inference
BINARY_V2_TOP_K_TOKENS = 15

# Backward compatibility (points to v1)
BINARY_MODEL_PATH = BINARY_V1_MODEL_PATH
BINARY_MAX_SEQ_LENGTH = BINARY_V1_MAX_SEQ_LENGTH
BINARY_ENABLE_ATTENTION = BINARY_V1_ENABLE_ATTENTION
BINARY_TOP_K_TOKENS = BINARY_V1_TOP_K_TOKENS

# Multi-Label Classifiers (3 setups)
MULTILABEL_SETUP1_PATH = "/home/avramit/classifier/01122025/classifier/multilabel/neodictabert-setup1_summarylevel-finetuned-multilabel_error_types"
MULTILABEL_SETUP2_V1_PATH = "/home/avramit/classifier/01122025/classifier/multilabel/neodictabert-setup2-claimlevel-finetuned-multilabel_error_types_021225"
MULTILABEL_SETUP2_V2_PATH = "/home/avramit/classifier/04122025/classifier/multilabel/neodictabert-setup2-claimlevel-finetuned-multilabel_error_types_041225"
MULTILABEL_SETUP3_PATH = "/home/avramit/classifier/01122025/classifier/multilabel/neodictabert-setup3-articleevidence_claimlevel-finetuned-multilabel_error_types_021225"
MULTILABEL_MAX_SEQ_LENGTH = 3072
MULTILABEL_THRESHOLD = 0.5  # Threshold for multi-label classification
MULTILABEL_ENABLE_ATTENTION = True  # Set to False for faster inference
MULTILABEL_TOP_K_TOKENS = 15

# Backward compatibility
MULTILABEL_SETUP2_PATH = MULTILABEL_SETUP2_V1_PATH

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
# Binary classifier v1 (using ORIGINAL working variable names)
binary_model = None
binary_tokenizer = None
binary_device = None
binary_config = None

# Binary classifier v2
binary_v2_model = None
binary_v2_tokenizer = None
binary_v2_device = None
binary_v2_config = None

# Multi-label classifiers
multilabel_setup1_model = None
multilabel_setup1_tokenizer = None
multilabel_setup1_config = None

multilabel_setup2_model = None
multilabel_setup2_tokenizer = None
multilabel_setup2_config = None

multilabel_setup2_v2_model = None
multilabel_setup2_v2_tokenizer = None
multilabel_setup2_v2_config = None

multilabel_setup3_model = None
multilabel_setup3_tokenizer = None
multilabel_setup3_config = None

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

class MultiLabelSetup1Request(BaseModel):
    article: str
    summary: str

class MultiLabelSetup2Request(BaseModel):
    article: str
    claim: str

class MultiLabelSetup3Request(BaseModel):
    evidence: str
    claim: str
    return_attention: bool = False

class MultiLabelResponse(BaseModel):
    predicted_labels: List[str]
    all_scores: dict
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
    global binary_v2_model, binary_v2_tokenizer, binary_v2_device, binary_v2_config
    global multilabel_setup1_model, multilabel_setup1_tokenizer, multilabel_setup1_config
    global multilabel_setup2_model, multilabel_setup2_tokenizer, multilabel_setup2_config
    global multilabel_setup2_v2_model, multilabel_setup2_v2_tokenizer, multilabel_setup2_v2_config
    global multilabel_setup3_model, multilabel_setup3_tokenizer, multilabel_setup3_config
    global llm_model, llm_tokenizer, llm_device

    # ========================================================================
    # BINARY CLASSIFIER V1
    # ========================================================================
    logger.info(f"Loading binary v1 model from {BINARY_V1_MODEL_PATH}...")
    try:
        # Load model first
        model_kwargs = {"trust_remote_code": True}
        if BINARY_V1_ENABLE_ATTENTION:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True

        binary_model = AutoModelForSequenceClassification.from_pretrained(
            BINARY_V1_MODEL_PATH,
            **model_kwargs
        )
        
        # Get config from loaded model
        binary_config = binary_model.config
        
        # Load tokenizer
        binary_tokenizer = AutoTokenizer.from_pretrained(BINARY_V1_MODEL_PATH, trust_remote_code=True)

        # Move to device
        binary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        binary_model.to(binary_device)
        binary_model.eval()

        logger.info(f"✅ Binary v1 classifier loaded on {binary_device}")
        if hasattr(binary_config, 'id2label') and hasattr(binary_config, 'num_labels'):
            logger.info(f"Labels: {[binary_config.id2label[i] for i in range(binary_config.num_labels)]}")
    except Exception as e:
        logger.error(f"❌ Failed to load binary v1 classifier: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # BINARY CLASSIFIER V2
    # ========================================================================
    logger.info(f"Loading binary v2 model from {BINARY_V2_MODEL_PATH}...")
    try:
        # Load model first
        model_kwargs = {"trust_remote_code": True}
        if BINARY_V2_ENABLE_ATTENTION:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True

        binary_v2_model = AutoModelForSequenceClassification.from_pretrained(
            BINARY_V2_MODEL_PATH,
            **model_kwargs
        )
        
        # Get config from loaded model
        binary_v2_config = binary_v2_model.config
        
        # Load tokenizer
        binary_v2_tokenizer = AutoTokenizer.from_pretrained(BINARY_V2_MODEL_PATH, trust_remote_code=True)

        # Move to device
        binary_v2_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        binary_v2_model.to(binary_v2_device)
        binary_v2_model.eval()

        logger.info(f"✅ Binary v2 classifier loaded on {binary_v2_device}")
        if hasattr(binary_v2_config, 'id2label') and hasattr(binary_v2_config, 'num_labels'):
            logger.info(f"Labels: {[binary_v2_config.id2label[i] for i in range(binary_v2_config.num_labels)]}")
    except Exception as e:
        logger.error(f"❌ Failed to load binary v2 classifier: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # MULTI-LABEL CLASSIFIERS (Setup 1, 2, 3)
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup 1: Article + Summary → Error Types
    logger.info(f"Loading multi-label setup1 from {MULTILABEL_SETUP1_PATH}...")
    try:
        multilabel_setup1_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILABEL_SETUP1_PATH, trust_remote_code=True
        )
        multilabel_setup1_config = multilabel_setup1_model.config
        multilabel_setup1_tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_SETUP1_PATH, trust_remote_code=True)
        multilabel_setup1_model.to(device)
        multilabel_setup1_model.eval()
        logger.info(f"✅ Multi-label setup1 loaded ({multilabel_setup1_config.num_labels} labels)")
    except Exception as e:
        logger.error(f"❌ Failed to load multi-label setup1: {e}")
        import traceback
        traceback.print_exc()
    
    # Setup 2 v1: Article + Claim → Error Types
    logger.info(f"Loading multi-label setup2 v1 from {MULTILABEL_SETUP2_V1_PATH}...")
    try:
        multilabel_setup2_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILABEL_SETUP2_V1_PATH, trust_remote_code=True
        )
        multilabel_setup2_config = multilabel_setup2_model.config
        multilabel_setup2_tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_SETUP2_V1_PATH, trust_remote_code=True)
        multilabel_setup2_model.to(device)
        multilabel_setup2_model.eval()
        logger.info(f"✅ Multi-label setup2 v1 loaded ({multilabel_setup2_config.num_labels} labels)")
    except Exception as e:
        logger.error(f"❌ Failed to load multi-label setup2 v1: {e}")
        import traceback
        traceback.print_exc()
    
    # Setup 2 v2: Article + Claim → Error Types
    logger.info(f"Loading multi-label setup2 v2 from {MULTILABEL_SETUP2_V2_PATH}...")
    try:
        multilabel_setup2_v2_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILABEL_SETUP2_V2_PATH, trust_remote_code=True
        )
        multilabel_setup2_v2_config = multilabel_setup2_v2_model.config
        multilabel_setup2_v2_tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_SETUP2_V2_PATH, trust_remote_code=True)
        multilabel_setup2_v2_model.to(device)
        multilabel_setup2_v2_model.eval()
        logger.info(f"✅ Multi-label setup2 v2 loaded ({multilabel_setup2_v2_config.num_labels} labels)")
    except Exception as e:
        logger.error(f"❌ Failed to load multi-label setup2 v2: {e}")
        import traceback
        traceback.print_exc()
    
    # Setup 3: Evidence + Claim → Error Types (with attention support)
    logger.info(f"Loading multi-label setup3 from {MULTILABEL_SETUP3_PATH}...")
    try:
        model_kwargs = {"trust_remote_code": True}
        if MULTILABEL_ENABLE_ATTENTION:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True
        
        multilabel_setup3_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILABEL_SETUP3_PATH, **model_kwargs
        )
        multilabel_setup3_config = multilabel_setup3_model.config
        multilabel_setup3_tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_SETUP3_PATH, trust_remote_code=True)
        multilabel_setup3_model.to(device)
        multilabel_setup3_model.eval()
        logger.info(f"✅ Multi-label setup3 loaded ({multilabel_setup3_config.num_labels} labels, attention={'enabled' if MULTILABEL_ENABLE_ATTENTION else 'disabled'})")
    except Exception as e:
        logger.error(f"❌ Failed to load multi-label setup3: {e}")
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
    
    # ========================================================================
    # STARTUP BANNER
    # ========================================================================
    print("\n" + "="*70)
    print("🚀 MULTI-MODEL INFERENCE API SERVER")
    print("="*70)
    print(f"📍 Server: http://{HOST}:{PORT}")
    print(f"💻 Device: {binary_device if binary_device else 'N/A'}")
    print("\n📦 LOADED MODELS:")
    if binary_model:
        print(f"  ✅ Binary Classifier v1: {BINARY_V1_MODEL_PATH}")
    if binary_v2_model:
        print(f"  ✅ Binary Classifier v2: {BINARY_V2_MODEL_PATH}")
    if multilabel_setup1_model:
        print(f"  ✅ Multi-Label Setup1: {MULTILABEL_SETUP1_PATH}")
    if multilabel_setup2_model:
        print(f"  ✅ Multi-Label Setup2 v1: {MULTILABEL_SETUP2_V1_PATH}")
    if multilabel_setup2_v2_model:
        print(f"  ✅ Multi-Label Setup2 v2: {MULTILABEL_SETUP2_V2_PATH}")
    if multilabel_setup3_model:
        print(f"  ✅ Multi-Label Setup3: {MULTILABEL_SETUP3_PATH}")
    if llm_model:
        print(f"  ✅ LLM Classifier: {LLM_MODEL_PATH} ({llm_device})")
    
    print("\n🌐 AVAILABLE ENDPOINTS:")
    if binary_model:
        print(f"  • POST http://{HOST}:{PORT}/binary/predict (v1)")
    if binary_v2_model:
        print(f"  • POST http://{HOST}:{PORT}/binary/v2/predict")
    if multilabel_setup1_model:
        print(f"  • POST http://{HOST}:{PORT}/multilabel/setup1/predict")
    if multilabel_setup2_model:
        print(f"  • POST http://{HOST}:{PORT}/multilabel/setup2/predict (v1)")
    if multilabel_setup2_v2_model:
        print(f"  • POST http://{HOST}:{PORT}/multilabel/setup2/v2/predict")
    if multilabel_setup3_model:
        print(f"  • POST http://{HOST}:{PORT}/multilabel/setup3/predict")
    if llm_model:
        print(f"  • POST http://{HOST}:{PORT}/llm/predict")
    print(f"  • GET  http://{HOST}:{PORT}/health")
    print("="*70 + "\n")

@app.get("/")
async def root():
    return {
        "status": "Multi-Model Inference API",
        "models": {
            "binary_v1": {
                "path": BINARY_V1_MODEL_PATH,
                "loaded": binary_model is not None
            },
            "binary_v2": {
                "path": BINARY_V2_MODEL_PATH,
                "loaded": binary_v2_model is not None
            },
            "multilabel_setup1": {
                "path": MULTILABEL_SETUP1_PATH,
                "loaded": multilabel_setup1_model is not None,
                "description": "Article + Summary → Error Types"
            },
            "multilabel_setup2_v1": {
                "path": MULTILABEL_SETUP2_V1_PATH,
                "loaded": multilabel_setup2_model is not None,
                "description": "Article + Claim → Error Types"
            },
            "multilabel_setup2_v2": {
                "path": MULTILABEL_SETUP2_V2_PATH,
                "loaded": multilabel_setup2_v2_model is not None,
                "description": "Article + Claim → Error Types"
            },
            "multilabel_setup3": {
                "path": MULTILABEL_SETUP3_PATH,
                "loaded": multilabel_setup3_model is not None,
                "description": "Evidence + Claim → Error Types"
            },
            "llm": {
                "path": LLM_MODEL_PATH,
                "loaded": llm_model is not None,
                "mode": "vLLM" if LLM_USE_VLLM else "unsloth"
            }
        },
        "binary_device": str(binary_device) if binary_device else None,
        "binary_v2_device": str(binary_v2_device) if binary_v2_device else None,
        "llm_device": str(llm_device) if llm_device else None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "binary_v1_model_loaded": binary_model is not None,
        "binary_v2_model_loaded": binary_v2_model is not None,
        "multilabel_setup1_loaded": multilabel_setup1_model is not None,
        "multilabel_setup2_v1_loaded": multilabel_setup2_model is not None,
        "multilabel_setup2_v2_loaded": multilabel_setup2_v2_model is not None,
        "multilabel_setup3_loaded": multilabel_setup3_model is not None,
        "llm_model_loaded": llm_model is not None,
        "binary_v1_device": str(binary_device) if binary_device else None,
        "binary_v2_device": str(binary_v2_device) if binary_v2_device else None,
        "llm_device": str(llm_device) if llm_device else None,
        # Backward compatibility
        "binary_model_loaded": binary_model is not None,
        "binary_device": str(binary_device) if binary_device else None,
        "multilabel_setup2_loaded": multilabel_setup2_model is not None
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
# BINARY CLASSIFIER V2 ENDPOINT
# ============================================================================
@app.post("/binary/v2/predict", response_model=BinaryClassifierResponse)
async def predict_binary_v2(request: BinaryClassifierRequest):
    """Run binary NLI inference on premise-hypothesis pair using v2 model"""

    if binary_v2_model is None or binary_v2_tokenizer is None:
        raise HTTPException(status_code=503, detail="Binary v2 model not loaded")

    try:
        inputs = binary_v2_tokenizer(
            request.premise,
            request.hypothesis,
            return_tensors="pt",
            padding="max_length",
            max_length=BINARY_V2_MAX_SEQ_LENGTH,
            truncation=True,
        )

        inputs = {k: v.to(binary_v2_device) for k, v in inputs.items()}

        with torch.no_grad():
            if BINARY_V2_ENABLE_ATTENTION and request.return_attention:
                outputs = binary_v2_model(**inputs, output_attentions=True)
            else:
                outputs = binary_v2_model(**inputs)

        logits = outputs.logits[0].cpu()
        probs = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class = binary_v2_config.id2label[predicted_class_idx]
        confidence = probs[predicted_class_idx].item()

        probabilities = {
            binary_v2_config.id2label[i]: float(probs[i].item())
            for i in range(binary_v2_config.num_labels)
        }
        
        # Normalize labels to match v1 format (entailment/contradiction)
        label_mapping = {
            "LABEL_0": "contradiction",
            "LABEL_1": "entailment"
        }
        
        # Map prediction if it's in LABEL_X format
        if predicted_class in label_mapping:
            predicted_class = label_mapping[predicted_class]
        
        # Map probabilities keys if they're in LABEL_X format
        if any(key in label_mapping for key in probabilities.keys()):
            probabilities = {
                label_mapping.get(key, key): value
                for key, value in probabilities.items()
            }

        attention_top_tokens = None
        if BINARY_V2_ENABLE_ATTENTION and request.return_attention and outputs.attentions is not None:
            attention_top_tokens = extract_attention_v2(inputs, outputs.attentions)

        return BinaryClassifierResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            attention_top_tokens=attention_top_tokens
        )

    except Exception as e:
        logger.error(f"Prediction error (v2): {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MULTI-LABEL CLASSIFIER ENDPOINTS
# ============================================================================
@app.post("/multilabel/setup1/predict", response_model=MultiLabelResponse)
async def predict_multilabel_setup1(request: MultiLabelSetup1Request):
    """Setup1: Article + Summary → Error Types"""
    
    if multilabel_setup1_model is None or multilabel_setup1_tokenizer is None:
        raise HTTPException(status_code=503, detail="Multi-label setup1 model not loaded")
    
    try:
        inputs = multilabel_setup1_tokenizer(
            request.article,
            request.summary,
            return_tensors="pt",
            padding="max_length",
            max_length=MULTILABEL_MAX_SEQ_LENGTH,
            truncation=True,
        )
        
        inputs = {k: v.to(binary_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = multilabel_setup1_model(**inputs)
        
        logits = outputs.logits[0].cpu()
        probs = torch.sigmoid(logits)
        
        # Get all scores
        all_scores = {
            multilabel_setup1_config.id2label[i]: float(probs[i].item())
            for i in range(multilabel_setup1_config.num_labels)
        }
        
        # Get predicted labels (above threshold)
        predicted_indices = (probs > MULTILABEL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [multilabel_setup1_config.id2label[i] for i in predicted_indices]
        
        return MultiLabelResponse(
            predicted_labels=predicted_labels,
            all_scores=all_scores
        )
    
    except Exception as e:
        logger.error(f"Multi-label setup1 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multilabel/setup2/predict", response_model=MultiLabelResponse)
async def predict_multilabel_setup2(request: MultiLabelSetup2Request):
    """Setup2: Article + Claim → Error Types"""
    
    if multilabel_setup2_model is None or multilabel_setup2_tokenizer is None:
        raise HTTPException(status_code=503, detail="Multi-label setup2 model not loaded")
    
    try:
        inputs = multilabel_setup2_tokenizer(
            request.article,
            request.claim,
            return_tensors="pt",
            padding="max_length",
            max_length=MULTILABEL_MAX_SEQ_LENGTH,
            truncation=True,
        )
        
        inputs = {k: v.to(binary_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = multilabel_setup2_model(**inputs)
        
        logits = outputs.logits[0].cpu()
        probs = torch.sigmoid(logits)
        
        # Get all scores
        all_scores = {
            multilabel_setup2_config.id2label[i]: float(probs[i].item())
            for i in range(multilabel_setup2_config.num_labels)
        }
        
        # Get predicted labels (above threshold)
        predicted_indices = (probs > MULTILABEL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [multilabel_setup2_config.id2label[i] for i in predicted_indices]
        
        return MultiLabelResponse(
            predicted_labels=predicted_labels,
            all_scores=all_scores
        )
    
    except Exception as e:
        logger.error(f"Multi-label setup2 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multilabel/setup2/v2/predict", response_model=MultiLabelResponse)
async def predict_multilabel_setup2_v2(request: MultiLabelSetup2Request):
    """Setup2 v2: Article + Claim → Error Types"""
    
    if multilabel_setup2_v2_model is None or multilabel_setup2_v2_tokenizer is None:
        raise HTTPException(status_code=503, detail="Multi-label setup2 v2 model not loaded")
    
    try:
        inputs = multilabel_setup2_v2_tokenizer(
            request.article,
            request.claim,
            return_tensors="pt",
            padding="max_length",
            max_length=MULTILABEL_MAX_SEQ_LENGTH,
            truncation=True,
        )
        
        inputs = {k: v.to(binary_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = multilabel_setup2_v2_model(**inputs)
        
        logits = outputs.logits[0].cpu()
        probs = torch.sigmoid(logits)
        
        # Get all scores
        all_scores = {
            multilabel_setup2_v2_config.id2label[i]: float(probs[i].item())
            for i in range(multilabel_setup2_v2_config.num_labels)
        }
        
        # Get predicted labels (above threshold)
        predicted_indices = (probs > MULTILABEL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [multilabel_setup2_v2_config.id2label[i] for i in predicted_indices]
        
        return MultiLabelResponse(
            predicted_labels=predicted_labels,
            all_scores=all_scores
        )
    
    except Exception as e:
        logger.error(f"Multi-label setup2 v2 prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multilabel/setup3/predict", response_model=MultiLabelResponse)
async def predict_multilabel_setup3(request: MultiLabelSetup3Request):
    """Setup3: Evidence + Claim → Error Types (with attention support)"""
    
    if multilabel_setup3_model is None or multilabel_setup3_tokenizer is None:
        raise HTTPException(status_code=503, detail="Multi-label setup3 model not loaded")
    
    try:
        inputs = multilabel_setup3_tokenizer(
            request.evidence,
            request.claim,
            return_tensors="pt",
            padding="max_length",
            max_length=MULTILABEL_MAX_SEQ_LENGTH,
            truncation=True,
        )
        
        inputs = {k: v.to(binary_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if MULTILABEL_ENABLE_ATTENTION and request.return_attention:
                outputs = multilabel_setup3_model(**inputs, output_attentions=True)
            else:
                outputs = multilabel_setup3_model(**inputs)
        
        logits = outputs.logits[0].cpu()
        probs = torch.sigmoid(logits)
        
        # Get all scores
        all_scores = {
            multilabel_setup3_config.id2label[i]: float(probs[i].item())
            for i in range(multilabel_setup3_config.num_labels)
        }
        
        # Get predicted labels (above threshold)
        predicted_indices = (probs > MULTILABEL_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [multilabel_setup3_config.id2label[i] for i in predicted_indices]
        
        # Extract attention if requested
        attention_top_tokens = None
        if MULTILABEL_ENABLE_ATTENTION and request.return_attention and outputs.attentions is not None:
            attention_top_tokens = extract_attention_multilabel(inputs, outputs.attentions, multilabel_setup3_tokenizer)
        
        return MultiLabelResponse(
            predicted_labels=predicted_labels,
            all_scores=all_scores,
            attention_top_tokens=attention_top_tokens
        )
    
    except Exception as e:
        logger.error(f"Multi-label setup3 prediction error: {e}")
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
    """Extract top K tokens by attention weight (content words only) - Binary classifier v1"""
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

        # Filter: keep only tokens with letters (exclude punctuation)
        filtered_results = []
        for idx, token in enumerate(content_tokens):
            if any(c.isalpha() for c in token):  # Has at least one letter
                filtered_results.append((idx, token, cls_attention_norm[idx]))
        
        # Sort by attention weight and take top K
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        top_k = min(BINARY_TOP_K_TOKENS, len(filtered_results))
        
        return [
            AttentionToken(
                token=token,
                weight=float(weight)
            )
            for _, token, weight in filtered_results[:top_k]
        ]

    except Exception as e:
        logger.error(f"Attention extraction error: {e}")
        return None


def extract_attention_v2(inputs, attentions) -> List[AttentionToken]:
    """Extract top K tokens by attention weight (content words only) - Binary classifier v2"""
    try:
        tokens = binary_v2_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

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

        # Filter: keep only tokens with letters (exclude punctuation)
        filtered_results = []
        for idx, token in enumerate(content_tokens):
            if any(c.isalpha() for c in token):  # Has at least one letter
                filtered_results.append((idx, token, cls_attention_norm[idx]))
        
        # Sort by attention weight and take top K
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        top_k = min(BINARY_V2_TOP_K_TOKENS, len(filtered_results))
        
        return [
            AttentionToken(
                token=token,
                weight=float(weight)
            )
            for _, token, weight in filtered_results[:top_k]
        ]

    except Exception as e:
        logger.error(f"Attention extraction error (v2): {e}")
        return None


def extract_attention_multilabel(inputs, attentions, tokenizer) -> List[AttentionToken]:
    """Extract top K tokens by attention weight (content words only) - Multi-label classifier"""
    try:
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

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

        # Filter: keep only tokens with letters (exclude punctuation)
        filtered_results = []
        for idx, token in enumerate(content_tokens):
            if any(c.isalpha() for c in token):  # Has at least one letter
                filtered_results.append((idx, token, cls_attention_norm[idx]))
        
        # Sort by attention weight and take top K
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        top_k = min(MULTILABEL_TOP_K_TOKENS, len(filtered_results))
        
        return [
            AttentionToken(
                token=token,
                weight=float(weight)
            )
            for _, token, weight in filtered_results[:top_k]
        ]

    except Exception as e:
        logger.error(f"Attention extraction error: {e}")
        return None

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
