import os
import re
from PIL import Image



def remove_corrupt_jpg_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not re.match(r'.*\.jpg', file):
                file_path = os.path.join(root, file)
                # os.remove(file_path)
                print('Removed', file_path)
    print('\nTotal removed:', count)



def is_jpg_corrupt(file):
    try:
        img = Image.open(file)
        img.verify()
    except (IOError, SyntaxError) as e:
        return True
    return False



def remove_corrupt_jpg_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(r'.*\.jpg', file):
                file_path = os.path.join(root, file)
                # print(file_path)
                if is_jpg_corrupt(file_path):
                    # os.remove(file_path)
                    count += 1
                    print('Removed', file_path)
    print('\nTotal removed:', count)



directory = "part-1"
remove_corrupt_jpg_files(directory)