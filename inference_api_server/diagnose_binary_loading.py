#!/usr/bin/env python3
"""
Diagnostic script to compare working vs failing binary classifier loading
"""

import sys
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"

print("="*80)
print("DIAGNOSTIC: Binary Classifier Loading")
print("="*80)

# Test 1: Check if transformers is patched
print("\n1. Check transformers version and patches:")
import transformers
print(f"   Transformers version: {transformers.__version__}")
print(f"   Transformers location: {transformers.__file__}")

# Test 2: Try loading config different ways
print("\n2. Loading config with AutoConfig.from_pretrained():")
try:
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"   ✅ Config loaded: {type(config)}")
    print(f"   Has model_type: {hasattr(config, 'model_type')}")
    if hasattr(config, 'model_type'):
        print(f"   model_type: {config.model_type}")
    print(f"   Config class: {config.__class__.__name__}")
    print(f"   Config dict keys: {list(vars(config).keys())[:10]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Load model first, then get config
print("\n3. Loading model first, then getting config from model:")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation="eager",
        output_attentions=True
    )
    config_from_model = model.config
    print(f"   ✅ Model loaded: {type(model)}")
    print(f"   Config from model: {type(config_from_model)}")
    print(f"   Has model_type: {hasattr(config_from_model, 'model_type')}")
    if hasattr(config_from_model, 'model_type'):
        print(f"   model_type: {config_from_model.model_type}")
    print(f"   Has id2label: {hasattr(config_from_model, 'id2label')}")
    if hasattr(config_from_model, 'id2label'):
        print(f"   Labels: {config_from_model.id2label}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check for unsloth interference
print("\n4. Check if unsloth is imported:")
if 'unsloth' in sys.modules:
    print(f"   ⚠️  UNSLOTH IS LOADED: {sys.modules['unsloth']}")
    print(f"   This may patch transformers and break NeoBERT!")
else:
    print(f"   ✅ Unsloth not loaded")

# Test 5: Check config.json directly
print("\n5. Check config.json file directly:")
import json
import os
config_file = os.path.join(MODEL_PATH, "config.json")
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    print(f"   ✅ config.json exists")
    print(f"   Type after json.load: {type(config_dict)}")
    print(f"   Has 'model_type': {'model_type' in config_dict}")
    if 'model_type' in config_dict:
        print(f"   model_type value: {config_dict['model_type']}")
else:
    print(f"   ❌ config.json not found")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

