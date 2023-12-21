import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def calc_bleu(reference_tokens, candidate_tokens):
    return sentence_bleu(reference_tokens, candidate_tokens)

def calc_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

# put original answers here
with open("D:/ITU/3rd_Semester/NLP/RAG_FLAN_T5/fine_tune_t5/answers.txt", "r") as ref_file:
    reference_lines = ref_file.readlines()

# put new answers here 
with open("D:/ITU/3rd_Semester/NLP/RAG_FLAN_T5/fine_tune_t5/fake_answers.txt", "r") as cand_file:
    candidate_lines = cand_file.readlines()

bleu_scores = {}
rouge1_scores = {}
rouge2_scores = {}
rougeL_scores = {}

# Calculate scores for each pair
for i in range(len(reference_lines)):
    reference = reference_lines[i].strip()
    candidate = candidate_lines[i].strip()

    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)

    bleu_score = calc_bleu(reference_tokens, candidate_tokens)
    rouge_scores = calc_rouge(reference, candidate)

    bleu_scores[i] = bleu_score
    rouge1_scores[i] = rouge_scores['rouge1'].fmeasure
    rouge2_scores[i] = rouge_scores['rouge2'].fmeasure
    rougeL_scores[i] = rouge_scores['rougeL'].fmeasure

    print(f"Pair {i + 1} - BLEU: {bleu_score}, ROUGE-1: {rouge_scores['rouge1'].fmeasure}, ROUGE-2: {rouge_scores['rouge2'].fmeasure}, ROUGE-L: {rouge_scores['rougeL'].fmeasure}")

# calculate score for each answer as well as  average scores
avg_bleu = sum(bleu_scores.values()) / len(bleu_scores)
avg_rouge1 = sum(rouge1_scores.values()) / len(rouge1_scores)
avg_rouge2 = sum(rouge2_scores.values()) / len(rouge2_scores)
avg_rougeL = sum(rougeL_scores.values()) / len(rougeL_scores)

print("\nAverage Scores:")
print(f"BLEU: {avg_bleu}")
print(f"ROUGE-1: {avg_rouge1}")
print(f"ROUGE-2: {avg_rouge2}")
print(f"ROUGE-L: {avg_rougeL}")