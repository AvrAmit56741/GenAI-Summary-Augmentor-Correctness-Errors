from math import exp

# Reference and candidate sentences
reference = "האימא עזרה בשיעורי הביית"
candidate = "האמא עזרה בשעורי הבית"

# Break into words
ref_words = ["האימא", "עזרה", "בשיעורי", "הביית"]
cand_words = ["האמא", "עזרה", "בשעורי", "הבית"]


# Calculate n-gram precisions
def calculate_ngram_precision(ref, cand, n):
    ref_ngrams = []
    cand_ngrams = []

    # Generate n-grams
    for i in range(len(ref) - n + 1):
        ref_ngrams.append(tuple(ref[i:i + n]))
    for i in range(len(cand) - n + 1):
        cand_ngrams.append(tuple(cand[i:i + n]))

    # Count matches
    matches = 0
    for ngram in cand_ngrams:
        if ngram in ref_ngrams:
            matches += 1

    if len(cand_ngrams) == 0:
        return 0
    return matches / len(cand_ngrams)


# Calculate precisions for n=1 to n=4
precisions = []
for n in range(1, 5):
    p = calculate_ngram_precision(ref_words, cand_words, n)
    precisions.append(p)

# Calculate brevity penalty
bp = 1  # Both sentences are same length, so BP = 1

# Calculate final BLEU score
# Using equal weights for all n-grams
weights = [0.25, 0.25, 0.25, 0.25]
weighted_score = sum(w * exp(1) if p == 0 else w * p for w, p in zip(weights, precisions))
bleu_score = bp * exp(weighted_score)

print("BLEU Score Analysis:")
print(f"Unigram precision: {precisions[0]:.3f}")
print(f"Bigram precision: {precisions[1]:.3f}")
print(f"Trigram precision: {precisions[2]:.3f}")
print(f"4-gram precision: {precisions[3]:.3f}")
print(f"Brevity Penalty: {bp}")
print(f"Final BLEU Score: {bleu_score:.3f}")