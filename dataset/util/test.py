from dataset import get_datapoint
import os
this_path = os.path.dirname(__file__)


split = "MLRC"
mode = "PC+refsol"
combined_ids = ["2303.11932_", "2309.05569_"] #["2110.03485_", "2309.05569_", "2303.11932_"]
for combined_id in combined_ids:
    X, y, metadata = get_datapoint(combined_id=combined_id, workspace=os.path.join(this_path, "../workspace"), verbose=True)
