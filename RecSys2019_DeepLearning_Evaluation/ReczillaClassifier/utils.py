from datetime import datetime

def print_special(str, logger):
    prnt = f"\n{datetime.now()}: {str}"
    print(prnt)
    logger.info(prnt)

def get_logger(logger_name):
    import logging
    logging.basicConfig(filename=f'ReczillaClassifier/logs/{logger_name}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger = logging.getLogger()
    return logger