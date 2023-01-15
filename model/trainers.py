from .city_transformer_trainer import CityTransformerTrainer
from .city_transformer_inverse_trainer import CityTransformerInverseTrainer

def get_trainer(name):
    TRAINERS = {
        'CityTransformer': CityTransformerTrainer,
        'CityTransformerInverse': CityTransformerInverseTrainer,
    }

    for n, t in TRAINERS.items():
        if n.lower() == name.lower():
            return t

    raise ValueError(f'trainer {name} is not defined')
