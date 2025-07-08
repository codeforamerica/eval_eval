import logging

"""
Simple logger implementation to provide CLI feedback.
"""

logger = logging.getLogger("se_eval_eval")
formatter = logging.Formatter("%(asctime)s: %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
