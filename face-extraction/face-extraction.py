import os
from pathlib import Path

import numpy as np
from PIL import Image
import face_recognition

from tqdm import tqdm



image_dir = Path('../dataset')
face_dir = Path('../faces')
os.makedirs(face_dir, exist_ok=True)



total_images = 0
for root, dirs, files in os.walk(image_dir):
    total_images += len(files)

print(f'Total Images: {total_images}.')



count = 1
for root, dirs, files in os.walk(image_dir):
# for root, dirs, files in tqdm(os.walk(image_dir)):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.webp'):

            if count%100 == 0:
                print(f"Processing Image: {count}/{total_images}")
            count += 1

            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert('RGB')

            ###################### RESIZE IMAGE #################################
            #####################################################################
            ##     aspect_ratio = img.width / img.height                       ##
            ##     max_size = (512, 512)                                       ##
            ##         if img.size[0] > img.size[1]:                           ##
            ##             new_width = max_size[0]                             ##
            ##             new_height = round(max_size[0] / aspect_ratio)      ##
            ##         else:                                                   ##
            ##             new_height = max_size[1]                            ##
            ##             new_width = round(max_size[1] * aspect_ratio)       ##
            ##         img = img.resize((new_width, new_height))               ##
            #####################################################################
            #####################################################################
            
            image = np.array(image)   
            face_box = face_recognition.face_locations(image)

            for i in range(len(face_box)):
                top, right, bottom, left = face_box[i]
                image_array = np.array(image)
                face_image = image_array[top+1:bottom, left+1:right]
                pil_image = Image.fromarray(face_image)
                
                base_filename, _ = os.path.splitext(filename)
                class_name = os.path.basename(root)
                class_dir = os.path.join(face_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                target_path = os.path.join(class_dir, f"img-{base_filename}-face-{i+1}.jpg")
                pil_image.save(target_path)