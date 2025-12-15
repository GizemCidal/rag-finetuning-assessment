import sys
import unittest
import os
from unittest.mock import MagicMock

# Add project root to sys.path to access the finetuning package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock Heavy Libraries
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["galore_torch"] = MagicMock()

# Mock specific attributes needed by the scripts
sys.modules["transformers"].AutoTokenizer.from_pretrained = MagicMock(return_value=MagicMock())
sys.modules["transformers"].AutoModelForCausalLM.from_pretrained = MagicMock(return_value=MagicMock())
sys.modules["datasets"].load_dataset = MagicMock(return_value=[{"instruction": "test", "output": "test"}])

print("[OK] Mocks initialized successfully.")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")
import finetuning
print(f"Finetuning package: {finetuning}")


def test_imports():
    print("Testing imports for 'finetuning.common'...")
    try:
        from finetuning.common import format_instruction, load_tokenizer_and_model
        print("[OK] finetuning.common imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Failed to import finetuning.common: {e}")
        return False

    print("Testing imports for 'finetuning.train_qlora'...")
    try:
        # Verify import safety
        import finetuning.train_qlora
        print("[OK] finetuning.train_qlora imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Failed to import finetuning.train_qlora: {e}")
        return False

    print("Testing imports for 'finetuning.train_galore'...")
    try:
        import finetuning.train_galore
        print("[OK] finetuning.train_galore imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Failed to import finetuning.train_galore: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if test_imports():
        print("\n[SUCCESS] REFACTOR VERIFICATION PASSED!")
        print("The code structure is correct and imports work as expected.")
    else:
        print("\n[FAILED] REFACTOR VERIFICATION FAILED!")
