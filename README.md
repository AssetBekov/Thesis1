# Thesis1

Image SR thesis 1 repository.

## Image SR

Image super-resolution is the process of obtaining the high-resolution image from corre-sponding low-resolution image
## Env preparation and Colab mounting

```bash
!git clone https://github.com/kkahloots/Generative_Models # this is for loading git with correct brach
from google.colab import drive
drive.mount('/content/drive')
!mkdir /content/drive/My\ Drive/Results
MAIN_SAVE_DIR = '/content/drive/My Drive/Results'
IMG_DIR = '/content/Generative_Models/data/.CBSD68' 
```
## Settings

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
import os
print(os.getcwd())
%env TF_KERAS = 1

sep_local = os.path.sep

import sys
sys.path.append('..'+sep_local+'..')
print(sep_local)
```

## Dataset Preparing and Convertion

```python
!git clone https://github.com/azeghost/LMDB_Datasets # github with dataset

!pip install colorlog
from colorlog import ColoredFormatter
images_dir = 'data/.CBSD68' #Folder to images ( without name of the folder which we moved images before)
validation_percentage = 30
valid_format = 'png'
transformer = SRLmdbTransformer(image_dir = images_dir, trans_func=shrink_fn,
                              validation_pct = validation_percentage, valid_image_formats = valid_format)
transformer.transform_store(labels_fn=get_label_by_filename,image_dir=images_dir, lmdb_dir = lmdb_dir
           ,category='training',target_size=(481, 321),color_mode='rgb')
transformer.transform_store(labels_fn=get_label_by_filename,image_dir=images_dir, lmdb_dir = lmdb_dir
           ,category='validation',target_size=(481, 321),color_mode='rgb')
```

## Results and Visualization 
*** Pictures ***

