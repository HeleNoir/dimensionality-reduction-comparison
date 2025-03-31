from pathlib import Path
import os

import utils


def view_log(path):
    """Read log file in path and convert it to csv"""
    file_path = Path(path).resolve()
    log = utils.read_log(file_path)
    file_name = os.path.basename(path)

    os.makedirs('data/logs', exist_ok=True)
    log.to_csv('data/logs/' + file_name + '.csv', index=False)
