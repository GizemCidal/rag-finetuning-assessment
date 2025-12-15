from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
"""
Evaluation module for RAG performance.

Calculates BLEU-4 and ROUGE-L scores to measure the similarity between generated answers
and ground truth.
"""

class Evaluator:
    """
    Evaluates RAG performance using BLEU and ROUGE metrics.
    """
    def __init__(self):
        """Initializes the evaluator with ROUGE scorer and smoothing function."""
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def calculate_metrics(self, reference: str, candidate: str):
        """
        Calculates BLEU and ROUGE scores.

        Args:
           reference (str): The ground truth text.
           candidate (str): The generated text.
        
        Returns:
            dict: Dictionary containing 'bleu4' and 'rouge_l' scores.
        """
        # BLEU-4
        # NLTK expects tokenized list for reference (list of lists) and candidate (list)
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # BLEU expects reference as list of lists
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
        
        # ROUGE-L
        rouge_scores = self.rouge.score(reference, candidate)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        return {
            "bleu4": bleu4,
            "rouge_l": rouge_l
        }
