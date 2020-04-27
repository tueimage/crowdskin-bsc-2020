from tensorflow.keras.preprocessing import image
import h5py
import os

IMAGE_DATA_PATH = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\data_bep\\isic-challenge-2017\\ISIC-2017_Training_Data\\'
os.chdir(IMAGE_DATA_PATH)
list_dir = os.listdir()[1:]
hf = h5py.File('all_images'+'.h5', 'w')
for filename in list_dir:
    cur_img = image.load_img(path=os.path.join(IMAGE_DATA_PATH, filename), grayscale=False,
                                         target_size=(384, 384))
    img = image.img_to_array(cur_img)
    hf.create_dataset(filename, data=img, compression='lzf')
hf.close()