from helpers.parser import parse
from model.trainers import get_trainer
import time

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name

    trainer = get_trainer(model_name)(**vars(args))
    trainer.initialize()

    ## Training or inference
    start = time.time()
    trainer.run()
    seconds = time.time() - start

    trainer.finalize(seconds=seconds)
