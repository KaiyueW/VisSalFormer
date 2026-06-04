from models.llava15 import LLaVA15
from models.qwen3vl import Qwen3VL
from models.bespoke import BespokeMinChart
from models.internvl import InternVL3
from models.chartr1 import ChartR1
 
MODELS = {
    "llava15":  LLaVA15,
    "qwen3vl":  Qwen3VL,
    "bespokeminchart": BespokeMinChart,
    "internvl": InternVL3,
    "chartr1": ChartR1,
}
 
def load_model(model_name: str):
    assert model_name in MODELS, f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
    return MODELS[model_name]().load()