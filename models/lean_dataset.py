from tensorflow.keras.preprocessing import image
import h5py
import os

# Definitions
IMAGE_DATA_PATH = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\data_bep\\isic-challenge-2017\\ISIC-2017_Training_Data\\'
TargetImageSize = (384, 384)

# Create smaller dataset with name "all_images" by applying resizing in advance for faster training

os.chdir(IMAGE_DATA_PATH)
list_dir = os.listdir()
clean_list_dir = [s for s in list_dir if 'jpg' in s]
hf = h5py.File('all_images'+'.h5', 'w')
for filename in clean_list_dir:
    cur_img = image.load_img(path=os.path.join(IMAGE_DATA_PATH, filename), grayscale=False,
                                         target_size=TargetImageSize)
    img = image.img_to_array(cur_img, dtype='u1')
    hf.create_dataset(filename, data=img, compression=None, dtype='u1')
hf.close()
