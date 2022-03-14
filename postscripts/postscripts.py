from ._city_transformer_postscripts import CityTransformerPostscripts
from ._city_transformer_inverse_postscripts import CityTransformerInversePostscripts

def get_postscripts(name):
    POST_SCRIPTS = {
        'CityTransformer': CityTransformerPostscripts,
        'CityTransformerInverse': CityTransformerInversePostscripts,
    }

    for n, p in POST_SCRIPTS.items():
        if n.lower() == name.lower():
            return p

    raise ValueError(f'trainer {name} is not defined')
