from dataset import get_datapoint, prepare_workspace
import os
this_path = os.path.dirname(__file__)

import argparse
import csv
import re
import json


split = "MLRC"
mode = "PC+refsol"
combined_ids = ["2205.00048_", "2303.11932_", "2309.05569_"]
for combined_id in combined_ids:
    X, y, metadata = get_datapoint(split, mode, combined_id, workspace=os.path.join(this_path, "../workspace"), verbose=True)