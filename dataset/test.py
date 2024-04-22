from dataset import get_datapoint
import os
this_path = os.path.dirname(__file__)

X, y = get_datapoint("PC+refsol", "0000.00000_0", workspace=os.path.join(this_path, "../", "workspace"), verbose=True)

print(X)
