import gdown
import os
from zipfile import ZipFile

if os.path.exists('src/BERT4Rec/src/outputs/pths/ml-1m_10Epochs'):
    print('Pretrained Model Already Downloaded ....')
else:
    print('Downloading Pretrained Models ....')
    url = 'https://drive.google.com/uc?id=1gordPPgI7sxo8ZxnRa02PQeaG4r69dBn'
    output = 'src/BERT4Rec/src/outputs/pths/'
    gdown.download(url, output, quiet = False)

    # loading the temp.zip and creating a zip object
    with ZipFile(output+'JME_extension.zip', 'r') as zObject:
        zObject.extractall(output)
    zObject.close()

    os.remove('src/BERT4Rec/src/outputs/pths/JME_extension.zip')
    print('Download of pretrained model completed')