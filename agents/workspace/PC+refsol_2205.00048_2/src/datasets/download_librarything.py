import gdown
import tarfile 
import os

if os.path.exists('src/datasets/lt/reviews.txt'):
    print('Dataset Librarything Already Downloaded ....')
else:
    print('Downloading LibraryThing Dataset')
    url = 'https://drive.google.com/u/0/uc?id=1wgXv14TyrRD5DH6ZfJqApFymQv7_vuho&export=download'
    output = 'src/datasets/lthing_data.tar.gz'
    gdown.download(url, output, quiet = False)

    mytar = tarfile.open(output)
    mytar.extractall('src/datasets/lt')
    mytar.close()

    os.rename('src/datasets/lt/lthing_data/reviews.txt', 'src/datasets/lt/reviews.txt')
    os.remove('src/datasets/lt/lthing_data/edges.txt')
    os.rmdir('src/datasets/lt/lthing_data')
    os.remove('src/datasets/lthing_data.tar.gz')
    print('Download of dataset LibraryThing completed')