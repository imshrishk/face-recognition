import gdown
import zipfile
import os


def get_download_link(URL):
    file_id = URL.split('/')[-2]
    download_link = f"https://drive.google.com/uc?id={file_id}"
    return download_link




# def download_dataset(URL, i):
#     URL = get_download_link(URL)
#     gdown.download(URL, output=f'data-{i}.zip', quiet=False)
    
#     # with zipfile.ZipFile(f'data-{i}.zip', 'r') as zip_ref:
#     #     zip_ref.extractall('.')
    
#     # os.remove(f'data-{i}.zip')


# URLs = [
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnWHJpX0t0XzhNSGM/view?usp=drive_link&resourcekey=0-KhDR2QuKg1Rry5Q3wusJKg", # part1.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnSjhCbWNlb1BXZWs/view?usp=drive_link&resourcekey=0-yVzlvxG007A4R6h7WPWKGw", # part2.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnazFWT2R2b1N5M1k/view?usp=drive_link&resourcekey=0-6IKcx-bXog0QdllPTFGpkA", # part3.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZncjl3ZXVrSGxrY28/view?usp=drive_link&resourcekey=0-pNqvGDu19eByEARmrHg1_w", # part4.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnUW1BZ2ZJWFNkZlU/view?usp=drive_link&resourcekey=0-RCGP25I14Q1vpx2482FpNg", # part5.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnQjJReE00UjhMZFE/view?usp=drive_link&resourcekey=0-PlvVeH22o4SmMTabCv8ouQ", # part6.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZna1ZYeW9YRHpNY1k/view?usp=drive_link&resourcekey=0-Sqzrvbu-KM5hplLzIZ4LmA", # part7.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnN0tPSi16RzYtMGM/view?usp=drive_link&resourcekey=0-D8q6Gbcfox2OvloC6WQ1mA", # part8.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnUUQxUVNDN19nQ0E/view?usp=drive_link&resourcekey=0-lA7VEZ0WAx7xuwmnFmu_jQ", # part9.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZncVVsOTBkNnJITHc/view?usp=drive_link&resourcekey=0-vHjrzFNxBLNiB2sceV5Jog", # part10.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnREFUS1FGZktUejg/view?usp=drive_link&resourcekey=0-RpkScc6MNRhEN_lQHb39ZQ", # part11.zip
#     "https://drive.google.com/file/d/0B5G8pYUQMNZnbjlVSVh6V21xcE0/view?usp=drive_link&resourcekey=0-dnDcZ5A9CrXfgLuv1c0sJQ"] # part12.zip


# for i, URL in enumerate(URLs):
#     download_dataset(URL, i+1)
#     print("\033[92mDownloaded data-{i+1}.zip\033[0m")

# print("\033[92mAll files downloaded successfully!\033[0m")