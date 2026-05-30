from models.llava15 import LLaVA15
# from models.internvl import InternVL
 
MODELS = {
    "llava15":  LLaVA15,
    #"internvl": InternVL,
}
 
def load_model(model_name: str):
    assert model_name in MODELS, f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
    return MODELS[model_name]().load()