from helpers.parser import parse
from postscripts.postscripts import get_postscripts
import time

if __name__ == '__main__':
    args = parse()
    model_name = args.model_name
    post_script = get_postscripts(model_name)(**vars(args))
    post_script.initialize()

    ## Convert data or save figures
    start = time.time()
    post_script.run()
    seconds = time.time() - start

    post_script.finalize(seconds=seconds)
