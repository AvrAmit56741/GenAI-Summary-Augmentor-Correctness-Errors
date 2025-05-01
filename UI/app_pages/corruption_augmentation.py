import streamlit as st
from transformers import pipeline
import random
import re


class Transformation:
    def transform(self, text, entities=None):
        pass


# Define LABEL_MAP if not already defined
LABEL_MAP = {True: "True", False: "False"}

class NegateSentences(Transformation):
    # Apply or remove negation from negatable tokens
    def __init__(self):
        super().__init__()
        # Define Hebrew negatable tokens (common verbs and auxiliaries)
        self.negatable_tokens = [
            "אהב", "לומד", "עושה", "יכול", "יכולה", "יכולים", "יכולות",
            "עשיתי", "עשית", "עושים", "עושות", "צריך", "צריכה",
            "צריכים", "צריכות", "ירצה", "ירצו", "ירצנה", "חייב",
            "חייבת", "חייבים", "חייבות", "עשה", "עשו"
        ]
        self.negation_particle = "לא"

    # CHANGED: Update the transform method signature to accept (text, entities=None)
    def transform(self, text, entities=None):
        new_claim, aug_span = self.__negate_sentences(text)

        if new_claim:
            return new_claim  # CHANGED: Return the modified text
        else:
            return text  # CHANGED: Return the original text if no negation applied

    def __negate_sentences(self, claim):
        # Tokenize the sentence into words and punctuation
        tokens = self.__simple_tokenize(claim)
        negated = False
        new_tokens = []
        aug_span = None

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if the current token is a negation particle
            if token == self.negation_particle:
                new_tokens.append(token)
                i += 1
                continue

            # Check if the token is a negatable verb
            base_token = self.__get_base_form(token)
            if base_token in self.negatable_tokens:
                # If "לא" is already before the verb, remove it (negation removal)
                if i > 0 and tokens[i - 1] == self.negation_particle:
                    new_tokens.pop()  # Remove "לא"
                    negated = True
                    aug_span = (i - 1, i)  # Span of "לא"
                else:
                    # Insert "לא" before the verb (negation addition)
                    new_tokens.append(self.negation_particle)
                    new_tokens.append(token)
                    negated = True
                    aug_span = (i, i + 1)  # Span of "לא"
                i += 1
                continue

            # If not negatable, keep the token as is
            new_tokens.append(token)
            i += 1

        if negated:
            # Reconstruct the sentence from tokens
            new_claim = self.__simple_detokenize(new_tokens)
            return new_claim, aug_span
        else:
            return None, None

    def __get_base_form(self, token):
        """
        A simplistic method to extract the base form of a Hebrew verb.
        This can be enhanced with more sophisticated morphological analysis if needed.
        """
        # Remove common Hebrew verb suffixes
        suffixes = ["ים", "ות", "ה", "ו", "ת", "תן", "תם", "נו", "נה"]
        for suffix in suffixes:
            if token.endswith(suffix):
                return token[:-len(suffix)]
        return token

    def __simple_tokenize(self, text):
        # Simple tokenizer: split by spaces and keep punctuation
        return re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)

    def __simple_detokenize(self, tokens):
        # Simple detokenizer: join tokens with spaces, handling punctuation
        text = ""
        for i, token in enumerate(tokens):
            if i > 0 and re.match(r'\w', token) and re.match(r'\w', tokens[i - 1]):
                text += " "
            text += token
        return text


class EntitySwap(Transformation):
    def transform(self, text, entities):
        # Filter entities to include only relevant types
        entity_candidates = [ent for ent in entities if ent["entity_group"] in ["B_PERS", "B_LOC", "B_ORG"]]
        if not entity_candidates:
            return text

        # Group entities by their type
        entities_by_type = {
            "B_PERS": [],
            "B_LOC": [],
            "B_ORG": []
        }
        for entity in entity_candidates:
            entities_by_type[entity["entity_group"]].append(entity)

        # Ensure we have replacement options for each type
        for entity_type in entities_by_type:
            if len(entities_by_type[entity_type]) < 2:  # Need at least two for replacement
                entities_by_type[entity_type] = []

        # Sort entities by start index in descending order
        sorted_entities = sorted(entity_candidates, key=lambda x: -x["start"])

        # Replace entities in the text
        new_text = text
        for entity in sorted_entities:
            entity_type = entity["entity_group"]
            replacements = [e for e in entities_by_type[entity_type] if e != entity]

            if replacements:  # Replace with a random other entity of the same type
                replacement_entity = random.choice(replacements)
                replacement = new_text[replacement_entity["start"]:replacement_entity["end"]]
            else:  # Fallback to the original hardcoded replacement if no options available
                replacement = "שם אחר" if entity_type == "B_PERS" else "מיקום אחר"

            # Apply the replacement with highlighting
            replacement = f'{replacement}'
            new_text = new_text[:entity["start"]] + replacement + new_text[entity["end"]:]

        return new_text





class NumberSwap(Transformation):
    def transform(self, text, entities=None):
        # Replace numbers with random numbers
        return re.sub(r'\d+', lambda x: str(random.randint(1, 100)), text)

import random
import re

class Transformation:
    def transform(self, text, entities=None):
        pass

class PronounSwap(Transformation):
    def __init__(self, prob_swap=0.5):  # CHANGED: Added prob_swap with a default value
        super().__init__()
        self.prob_swap = prob_swap
        # Mapping from pronoun "classes" to actual Hebrew pronouns
        self.class2pronoun_map = {
            "SUBJECT": ["אני", "אתה", "את", "הוא", "היא", "אנחנו", "אתם", "אתן", "הם", "הן"],
            "OBJECT": ["אותי", "אותך", "אותו", "אותה", "אותנו", "אתכם", "אתכן", "אותם", "אותן"],
            "POSSESSIVE": ["שלי", "שלך", "שלו", "שלה", "שלנו", "שלכם", "שלכן", "שלהם", "שלהן"],
            "REFLEXIVE": ["עצמך", "עצמכם", "עצמכן", "עצמם", "עצמן"]
        }

        # Inverse mapping: for each pronoun, find which class it belongs to
        self.pronoun2class_map = {
            pronoun: category
            for category, pronouns in self.class2pronoun_map.items()
            for pronoun in pronouns
        }

        # A set of all known pronouns (for quick membership checks)
        self.pronouns = set(self.pronoun2class_map.keys())

    def transform(self, text, entities=None):
        words = text.split()
        # 1. Collect all occurrences (index, word) of pronouns
        pronoun_positions = []
        for i, word in enumerate(words):
            if word in self.pronouns:
                pronoun_positions.append((i, word))

        # If there are no pronouns, nothing to swap
        if not pronoun_positions:
            return text

        # 2. Randomly choose exactly one pronoun to swap
        index_to_swap, original_pronoun = random.choice(pronoun_positions)
        pronoun_class = self.pronoun2class_map[original_pronoun]

        # 3. Choose a new pronoun from the same class, but different from the original
        replacements = [p for p in self.class2pronoun_map[pronoun_class] if p != original_pronoun]
        if not replacements:
            # No valid replacements; return the original text
            return text

        new_pronoun = random.choice(replacements)

        # 4. Replace the chosen pronoun in-place and highlight it
        words[index_to_swap] = f'<span style="background-color: #add8e6; color: black;">{new_pronoun}</span>'

        return " ".join(words)


class AddNoise(Transformation):
    # Inject noise into Hebrew claims
    def __init__(self, noise_prob=0.05, delete_prob=0.8):
        super().__init__()

        self.noise_prob = noise_prob
        self.delete_prob = delete_prob
        # Removed SpaCy dependency as Hebrew support is not available

    # CHANGED: Update the transform method signature to accept (text, entities=None)
    def transform(self, text, entities=None):
        new_claim, aug_span = self.__add_noise(text, entities)

        if new_claim and new_claim != text:
            # Optionally, you can highlight the noise or track it
            return new_claim  # CHANGED: Return the modified text
        else:
            return text  # CHANGED: Return the original text if no noise injected

    def __add_noise(self, claim, aug_span):
        tokens = self.__simple_tokenize(claim)
        new_tokens = []
        updated_aug_span = aug_span

        for ix, token in enumerate(tokens):
            # Handle augmented span check properly
            apply_augmentation = True
            if isinstance(updated_aug_span, tuple):
                span_start, span_end = updated_aug_span
                if span_start <= ix <= span_end:
                    apply_augmentation = False

            if apply_augmentation and random.random() < self.noise_prob:
                if random.random() < self.delete_prob:
                    if isinstance(updated_aug_span, tuple):
                        span_start, span_end = updated_aug_span
                        if ix < span_start:
                            span_start -= 1
                            span_end -= 1
                            updated_aug_span = (span_start, span_end)
                    continue
                else:
                    new_tokens.append(token)
            new_tokens.append(token)

        new_claim = self.__simple_detokenize(new_tokens)
        return (new_claim, updated_aug_span) if new_claim != claim else (None, None)

    def __simple_tokenize(self, text):
        # Simple tokenizer: split by spaces and keep punctuation
        return re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)

    def __simple_detokenize(self, tokens):
        # Simple detokenizer: join tokens with spaces, handling punctuation
        text = ""
        for i, token in enumerate(tokens):
            if i > 0:
                # Add space before the token if both current and previous tokens are words
                if re.match(r'\w', token) and re.match(r'\w', tokens[i - 1]):
                    text += " "
                # No space before punctuation
                elif re.match(r'[^\w\s]', token):
                    pass
                else:
                    text += " "
            text += token
        return text


def split_text_into_chunks(text, max_tokens=510, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks


def truncate_text(text, tokenizer, max_length=512):
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

def augment_text(text, transformations, max_tokens=512):
    chunks = split_text_into_chunks(text, max_tokens=max_tokens)

    ner_model = pipeline(
        "token-classification",
        model="avichr/heBERT_NER",
        tokenizer="avichr/heBERT_NER",
        aggregation_strategy="simple",
    )

    augmented_text = ""
    for chunk in chunks:
        chunk = truncate_text(chunk, ner_model.tokenizer)
        entities = ner_model(chunk) if any(isinstance(t, EntitySwap) for t in transformations) else None

        for transformation in transformations:
            chunk = transformation.transform(chunk, entities)

        augmented_text += chunk + " "
    return augmented_text.strip()


def app():
    st.title("Hebrew Text Augmentation")
    st.write("Enter Hebrew text and select augmentation methods to view the modified text.")

    input_text = st.text_area("Input Hebrew Text", placeholder="לדוגמה: אני אוהב ללמוד בתל אביב עם החברים שלי.")

    entity_swap_selected = st.checkbox("Entity Swap (using NER model of avichr/heBERT_NER, support entity_groups: [B_PERS, B_LOC, B_ORG])", value=True)
    number_swap_selected = st.checkbox("Number Swap (change numbers)")
    pronoun_swap_selected = st.checkbox("Pronoun Swap (change pronouns)")
    negate_sentences_selected = st.checkbox("Negate Sentences (הפיכת משפט לשלילה)")
    add_noise_selected = st.checkbox("Add Noise, mimic model generation noise")  # Added checkbox for AddNoise

    if st.button("Augment Text"):
        if not input_text.strip():
            st.warning("Please enter some text.")
            return

        transformations = []
        if entity_swap_selected:
            transformations.append(EntitySwap())
        if number_swap_selected:
            transformations.append(NumberSwap())
        if pronoun_swap_selected:
            transformations.append(PronounSwap(prob_swap=0.99))
        if negate_sentences_selected:
            transformations.append(NegateSentences())
        if add_noise_selected:
            transformations.append(AddNoise(noise_prob=0.01, delete_prob=0.99))  # Append AddNoise

        augmented_text = augment_text(input_text, transformations)

        st.subheader("Augmented Text")
        st.markdown(augmented_text, unsafe_allow_html=True)  # Enable HTML rendering



if __name__ == "__main__":
    app()
