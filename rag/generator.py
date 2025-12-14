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

    def generate_answer(self, query: str, context: str, do_sample: bool = True) -> str:
        """
        Generates an answer given the query and context.
        Args:
            query: The user question.
            context: The retrieved context.
            do_sample: If True, uses random sampling (creative). If False, uses greedy decoding (deterministic/eval).
        """
        
        # System Prompt for Safety & Grounding
        system_prompt = (
            "You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context below. "
            "If the context does not contain the answer, politely state that you cannot answer based on the available information."
        )
        
        # Construct Prompt using Chat Template
        # Note: Gemma chat template might handle system roles differently, but usually 'user' with instructions works well.
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generation config
        if do_sample:
            gen_kwargs = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        else:
            # Deterministic for Evaluation
            gen_kwargs = {
                "do_sample": False,
                "temperature": 0.0, # Greedy
            }

        outputs = self.pipe(prompt, max_new_tokens=256, **gen_kwargs)
        
        generated_text = outputs[0]["generated_text"]
        
        # Extract only the model's response (remove the prompt)
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text
            
        return answer
