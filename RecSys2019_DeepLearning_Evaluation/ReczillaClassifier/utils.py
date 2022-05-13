from datetime import datetime
import logging

logging.basicConfig(filename='classifier.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

def print_special(str):
    prnt = f"\n{datetime.now()}: {str}"
    print(prnt)
    logger.info(prnt)