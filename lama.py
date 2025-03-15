from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np

import os
from tqdm import tqdm

simple_lama = SimpleLama()
img_path = "open/test_input"
mask_path = "open/test_mask"

img_name = os.listdir(img_path)
mask_name = os.listdir(mask_path)
count = len(img_name)

for i in tqdm(range(count)):
    imge = Image.open(img_path + '/' + img_name[i]).convert('RGB')
    imge = np.array(imge)
    mask = np.load(mask_path + '/' + img_name[i].split('.')[0] + '.npy')

    result = simple_lama(imge, mask)
    result.save("open/test_output/mask/" + img_name[i])
