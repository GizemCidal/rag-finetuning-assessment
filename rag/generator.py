import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import RAGConfig

class RAGGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        print(f"Loading LLM: {self.config.LLM_MODEL_NAME}")
        
        # Optimize for Colab: using minimal types if possible, or mapping to auto device
        # Note: gemma-3-1b-it might require authentication or agreeing to terms.
        # Ensure the user is logged in via huggingface-cli login if this is a gated model.
        # Assuming standard access or open model for this task.
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL_NAME, 
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer given the query and context."""
        
        # Simple RAG Prompt
        prompt_template = f"""<start_of_turn>user
You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question:
{query}<end_of_turn>
<start_of_turn>model
"""
        # Note: Gemma uses specific chat templates. The above is a raw approximation or we can use apply_chat_template.
        # Let's use clean apply_chat_template for best results.
        
        messages = [
            {"role": "user", "content": f"Answer the question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)
        
        generated_text = outputs[0]["generated_text"]
        
        # Extract only the model's response (remove the prompt)
        # Usually pipeline returns the full text.
        # We can strip the prompt.
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text
            
        return answer
