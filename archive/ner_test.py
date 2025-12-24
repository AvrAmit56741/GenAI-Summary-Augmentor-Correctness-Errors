# from transformers import pipeline
# import random
#
#
# # Base class for transformations
# class Transformation:
#     def transform(self, text, entities):
#         pass
#
#
# # EntitySwap class for Hebrew
# class EntitySwap(Transformation):
#     def transform(self, text, entities):
#         entity_candidates = [ent for ent in entities if ent["entity_group"] in ["B_PERS", "B_ORG", "B_LOC"]]
#         if not entity_candidates:
#             return text
#
#         entity_to_replace = random.choice(entity_candidates)
#         replacement = "מיקום אחר" if entity_to_replace["entity_group"] == "B_LOC" else "שם אחר"
#         new_text = text[: entity_to_replace["start"]] + replacement + text[entity_to_replace["end"] :]
#         return new_text
#
#
# # NumberSwap class
# class NumberSwap(Transformation):
#     def transform(self, text, entities):
#         number_candidates = [ent for ent in entities if ent["entity_group"] in ["B_MONEY", "B_PERCENT"]]
#         if not number_candidates:
#             return text
#
#         number_to_replace = random.choice(number_candidates)
#         replacement = str(random.randint(1, 100)) if number_to_replace["entity_group"] == "B_PERCENT" else "סכום אחר"
#         new_text = text[: number_to_replace["start"]] + replacement + text[number_to_replace["end"] :]
#         return new_text
#
#
# # DateSwap class
# class DateSwap(Transformation):
#     def transform(self, text, entities):
#         date_candidates = [ent for ent in entities if ent["entity_group"] == "B_DATE"]
#         if not date_candidates:
#             return text
#
#         date_to_replace = random.choice(date_candidates)
#         replacement = "11/09/1995"
#         new_text = text[: date_to_replace["start"]] + replacement + text[date_to_replace["end"] :]
#         return new_text
#
#
# # Main augmentation function
# def augment_text(text, transformations):
#     ner_model = pipeline(
#         "token-classification",
#         model="avichr/heBERT_NER",
#         tokenizer="avichr/heBERT_NER",
#         aggregation_strategy="simple",  # Group tokens into entities
#     )
#     entities = ner_model(text)
#     print(f"Detected Entities: {entities}")
#
#     augmented_text = text
#     for transformation in transformations:
#         augmented_text = transformation.transform(augmented_text, entities)
#     return augmented_text
#
#
# # Example usage
# if __name__ == "__main__":
#     input_text = "דויד עובד בארגון ידוע בתל אביב ביום 15/08/2024."
#     print(f"Original Text: {input_text}")
#
#     transformations = [EntitySwap(), NumberSwap(), DateSwap()]
#     augmented_text = augment_text(input_text, transformations)
#
#     print(f"Augmented Text: {augmented_text}")
# from ragas.dataset_schema import SingleTurnSample
# from ragas.metrics._factual_correctness import FactualCorrectness
#
#
# sample = SingleTurnSample(
#     response="The Eiffel Tower is located in Paris.",
#     reference="The Eiffel Tower is located in Paris. I has a height of 1000ft."
# )
#
# scorer = FactualCorrectness()
# #scorer.llm = openai_model
# #await scorer.single_turn_ascore(sample)
# #scorer.llm = openai_model
# #await scorer.single_turn_ascore(sample)
