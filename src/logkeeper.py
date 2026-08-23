import logging
from datetime import datetime


def setup_logging(name, level = logging.INFO):
    log_filename = datetime.now().strftime(name+"_%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(filename=log_filename, 
                        level=level, 
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    return log_filename