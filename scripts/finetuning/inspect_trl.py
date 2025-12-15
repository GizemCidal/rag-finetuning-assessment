from trl import SFTTrainer, SFTConfig
import inspect

print("SFTTrainer init args:", inspect.signature(SFTTrainer.__init__))
print("SFTConfig init args:", inspect.signature(SFTConfig.__init__))
