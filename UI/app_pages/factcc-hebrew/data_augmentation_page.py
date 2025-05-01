import streamlit as st
import json
from io import BytesIO
from app_pages.factcc.augmentation_ops import (
    SampleSentences,
    NegateSentences,
    PronounSwap,
    EntitySwap,
    NumberSwap,
    DateSwap,
    AddNoise
)

# We remove "Sample Sentences" from the list the user can choose,
# because we ALWAYS apply it automatically in code.
TRANSFORMATIONS = {
    "Negate Sentences - Semantically Invariant": NegateSentences(),
    "Pronoun Swap - Semantically Variant": PronounSwap(),
    "Entity Swap - Semantically Variant": EntitySwap(),
    "Number Swap - Semantically Variant": NumberSwap(),
    "Date Swap - Semantically Variant": DateSwap(),
    "Add Noise - Semantically Invariant": AddNoise(),
    "Paraphrasing - Semantically Invariant": DateSwap() #changed!
}

def app():
    st.title("Synthetic Data Augmentation for FactCC")

    # Document input fields
    st.header("Input Document Information")
    doc_id = st.text_input("Document ID")
    doc_text = st.text_area("Document Text")

    st.write(
        """
        **Instructions:**  
        1. Enter a Document ID and the source document text.  
        2. We will automatically pick (sample) one sentence from your document 
           as the initial claim.  
        3. Choose any *additional* transformations to apply on the claim.  
        4. Click "Perform Augmentation" to generate augmented claims.
        """
    )

    # Transformation selection (excluding SampleSentences, which is applied automatically)
    st.header("Choose Augmentation Methods (Optional)")
    selected_transformations = st.multiselect(
        "Select transformations to apply AFTER sampling a sentence:",
        TRANSFORMATIONS.keys(),
    )

    # Perform augmentation button
    if st.button("Perform Augmentation"):
        if not doc_id or not doc_text.strip():
            st.error("Please provide both a Document ID and some Document Text.")
        else:
            # 1) Convert text to a spaCy Doc (using the same spaCy model as in SampleSentences)
            spacy_model = SampleSentences().spacy
            doc = spacy_model(doc_text)

            # 2) Create a base example with no claim initially
            base_example = {
                "id": doc_id,
                "text": doc,   # store the full spaCy Doc
                "claim": None, # will be populated by SampleSentences
                "label": None,
                "extraction_span": None,
                "backtranslation": False,
                "augmentation": None,
                "augmentation_span": None,
                "noise": False,
            }

            # 3) Always apply SampleSentences transformation to pick a sentence
            sampler = SampleSentences()

            base_example["text"] = base_example["text"].text if hasattr(base_example["text"], "text") else base_example["text"]
            example_with_claim = sampler.transform(base_example)
            example_with_claim["text"] = example_with_claim["text"].text if hasattr(example_with_claim["text"],"text") else example_with_claim["text"]

            if not example_with_claim or not example_with_claim.get("claim"):
                st.warning("Could not sample a sentence from the document. Please check your text.")
                return

            # 4) Apply any user-selected transformations
            augmented_examples = []
            for transformation_name in selected_transformations:
                operation = TRANSFORMATIONS[transformation_name]
                example_with_claim["text"] = spacy_model(example_with_claim["text"]) if isinstance(
                    example_with_claim["text"], str) else example_with_claim["text"]
                new_example = operation.transform(example_with_claim)

                if new_example:
                    augmented_examples.append(new_example)

            # If the user didn't pick any transformations, let's at least include
            # the example_with_claim in our final output (the "sampled only" result).
            if not selected_transformations:
                augmented_examples.append(example_with_claim)

            # 5) Write augmented examples to a JSONL file
            if augmented_examples:
                output_file = BytesIO()
                for ex in augmented_examples:
                    # Convert spaCy Doc objects back to string text
                    ex["text"] = ex["text"] if isinstance(ex["text"], str) else ex["text"].text
                    ex["claim"] = ex["claim"] if isinstance(ex["claim"], str) else ex["claim"].text

                    line = json.dumps(ex, ensure_ascii=False)
                    output_file.write(line.encode("utf-8") + b"\n")
                output_file.seek(0)

                st.success(f"Augmentation completed. {len(augmented_examples)} examples generated.")
                st.download_button(
                    label="Download Augmented Data",
                    data=output_file,
                    file_name=f"{doc_id}_augmented.jsonl",
                    mime="application/jsonl",
                )
            else:
                st.warning("No augmented examples were generated.")
