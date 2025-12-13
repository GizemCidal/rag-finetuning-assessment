from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def evaluate(self, candidate: str, reference: str):
        # BLEU-4
        # NLTK expects tokenized list for reference (list of lists) and candidate (list)
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # Note: reference is list of lists
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
        
        # ROUGE-L
        rouge_scores = self.rouge.score(reference, candidate)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        return {
            "bleu": bleu4,
            "rouge": rouge_l
        }
