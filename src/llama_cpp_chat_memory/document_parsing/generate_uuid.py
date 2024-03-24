import logging
import uuid

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)

logging.debug(str(uuid.uuid1()))
