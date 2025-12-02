"""
Diagnostic script to check if a model supports attention output.
This helps identify why attention visualization might not work.
"""

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"

# ============================================================================


def diagnose_model_attention(model_path: str):
    """
    Diagnose whether a model can output attention weights.
    """
    print("=" * 70)
    print("MODEL ATTENTION DIAGNOSTIC")
    print("=" * 70)
    print(f"\nModel Path: {model_path}\n")
    
    # ========================================
    # 1. Check config.json
    # ========================================
    print("=" * 70)
    print("1. MODEL CONFIGURATION")
    print("=" * 70)
    
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"\nModel Type: {config.model_type}")
        print(f"Architecture: {config.architectures}")
        print(f"Number of Labels: {config.num_labels}")
        print(f"Hidden Size: {config.hidden_size}")
        print(f"Number of Layers: {config.num_hidden_layers}")
        print(f"Number of Attention Heads: {config.num_attention_heads}")
        
        # Check attention-related config
        print(f"\n--- Attention Configuration ---")
        print(f"output_attentions: {getattr(config, 'output_attentions', 'NOT SET')}")
        print(f"output_hidden_states: {getattr(config, 'output_hidden_states', 'NOT SET')}")
        
        # Check if config has attention dropout (indicates attention mechanism exists)
        print(f"attention_probs_dropout_prob: {getattr(config, 'attention_probs_dropout_prob', 'NOT SET')}")
        
        # Display id2label mapping
        print(f"\n--- Label Mapping ---")
        if hasattr(config, 'id2label'):
            for idx, label in config.id2label.items():
                print(f"  {idx}: {label}")
        else:
            print("  No id2label mapping found")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # ========================================
    # 2. Load model WITHOUT attention
    # ========================================
    print("\n" + "=" * 70)
    print("2. LOADING MODEL (output_attentions=False)")
    print("=" * 70)
    
    try:
        model_no_attn = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            output_attentions=False
        )
        print(f"✅ Model loaded successfully")
        print(f"Model class: {type(model_no_attn).__name__}")
        print(f"Model has {sum(p.numel() for p in model_no_attn.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # ========================================
    # 3. Load model WITH attention
    # ========================================
    print("\n" + "=" * 70)
    print("3. LOADING MODEL (output_attentions=True)")
    print("=" * 70)
    
    try:
        model_with_attn = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="eager",  # Force eager attention for NeoBERT
            output_attentions=True
        )
        print(f"✅ Model loaded successfully with output_attentions=True and eager attention")
        
        # Check if config was updated
        if hasattr(model_with_attn.config, 'output_attentions'):
            print(f"Config.output_attentions: {model_with_attn.config.output_attentions}")
        else:
            print("⚠️  Config does not have output_attentions attribute")
        
    except Exception as e:
        print(f"❌ Error loading model with attention: {e}")
        return
    
    # ========================================
    # 4. Test inference with dummy input
    # ========================================
    print("\n" + "=" * 70)
    print("4. TEST INFERENCE WITH ATTENTION")
    print("=" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Create dummy input
        dummy_text1 = "This is a test premise."
        dummy_text2 = "This is a test hypothesis."
        
        inputs = tokenizer(
            dummy_text1,
            dummy_text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        print(f"\nTest input created:")
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Tokens: {inputs['input_ids'].shape[1]}")
        
        # Run inference
        model_with_attn.eval()
        with torch.no_grad():
            # Try passing output_attentions explicitly in forward pass
            outputs = model_with_attn(**inputs, output_attentions=True)
        
        print(f"\n✅ Inference successful!")
        print(f"\nOutput attributes:")
        for attr in dir(outputs):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        
        # Check attentions
        print(f"\n--- Attention Check ---")
        if hasattr(outputs, 'attentions'):
            if outputs.attentions is not None:
                print(f"✅ outputs.attentions exists: {type(outputs.attentions)}")
                print(f"   Number of layers: {len(outputs.attentions)}")
                if len(outputs.attentions) > 0:
                    print(f"   Last layer shape: {outputs.attentions[-1].shape}")
                    print(f"   Expected: (batch=1, heads={config.num_attention_heads}, seq_len, seq_len)")
            else:
                print(f"❌ outputs.attentions is None")
        else:
            print(f"❌ outputs does not have 'attentions' attribute")
        
        # Check hidden states
        print(f"\n--- Hidden States Check ---")
        if hasattr(outputs, 'hidden_states'):
            if outputs.hidden_states is not None:
                print(f"✅ outputs.hidden_states exists: {type(outputs.hidden_states)}")
                print(f"   Number of layers: {len(outputs.hidden_states)}")
            else:
                print(f"❌ outputs.hidden_states is None")
        else:
            print(f"❌ outputs does not have 'hidden_states' attribute")
        
        # Check logits
        print(f"\n--- Logits Check ---")
        if hasattr(outputs, 'logits'):
            print(f"✅ outputs.logits shape: {outputs.logits.shape}")
            print(f"   Expected: (batch=1, num_labels={config.num_labels})")
        else:
            print(f"❌ outputs does not have 'logits' attribute")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # 5. Model Architecture Inspection
    # ========================================
    print("\n" + "=" * 70)
    print("5. MODEL ARCHITECTURE INSPECTION")
    print("=" * 70)
    
    print(f"\nModel modules:")
    for name, module in model_with_attn.named_modules():
        if 'attention' in name.lower():
            print(f"  {name}: {type(module).__name__}")
    
    # Check if base model has attention
    if hasattr(model_with_attn, 'bert'):
        print(f"\n✅ Model has 'bert' attribute (BERT-based)")
    elif hasattr(model_with_attn, 'roberta'):
        print(f"\n✅ Model has 'roberta' attribute (RoBERTa-based)")
    elif hasattr(model_with_attn, 'distilbert'):
        print(f"\n✅ Model has 'distilbert' attribute (DistilBERT-based)")
    else:
        print(f"\n⚠️  Unknown base model type")
    
    # ========================================
    # 6. Summary and Recommendations
    # ========================================
    print("\n" + "=" * 70)
    print("6. DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    can_output_attention = (
        hasattr(outputs, 'attentions') and 
        outputs.attentions is not None and 
        len(outputs.attentions) > 0
    )
    
    if can_output_attention:
        print("\n✅ GOOD NEWS: Your model CAN output attention weights!")
        print("\nAttention is working correctly. If you still see issues in the inference script:")
        print("  1. Make sure output_attentions=True is set when loading the model")
        print("  2. Check that you're using model(**inputs) not model.forward()")
        print("  3. Verify the inference script is checking outputs.attentions correctly")
    else:
        print("\n❌ ISSUE DETECTED: Model cannot output attention weights")
        print("\nPossible reasons:")
        print("  1. Model was fine-tuned with a modified architecture that removed attention")
        print("  2. Model weights don't include attention layers")
        print("  3. Model type doesn't support output_attentions parameter")
        print("  4. Transformers version incompatibility")
        
        print("\n💡 SOLUTIONS:")
        print("  Option 1: Set ENABLE_ATTENTION_ANALYSIS=False in inference script")
        print("  Option 2: Re-train the model without modifying the architecture")
        print("  Option 3: Load the original base model for attention visualization")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    diagnose_model_attention(MODEL_PATH)

