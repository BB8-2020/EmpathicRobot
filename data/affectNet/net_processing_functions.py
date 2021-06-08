import os
import cv2
import numpy as np
import matplotlib.image as mpimg


def preprocess_data(data):
    """Resize the images based on what size de model is trained on. Turn the needed columns into numpy arrays."""
    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(data['target'])
    X = np.zeros((n_samples, w, h, 3))
    
    for i in range(n_samples):
        X[i] = cv2.resize(data['formatted_pixels'].iloc[i], dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        
    return X, y


def clean_data_and_normalize(X, y):
    """Normalize the data."""
    # Normalize image vectors
    X = X / 255.0

    return X, y


def get_latest_index():
    """Get index of the last image in folder."""
    latest_img = 0
    arr = os.listdir('train_set/images')

    for i in arr:
        index = i.split('.')[0]
        if int(index) > latest_img:
            latest_img = int(index)

    return latest_img
    

def convert_to_dataframe(latest_img, df, emotions, path):
    """Loop through all images, except for datasets larger than the wanted size.
    Load these images with labels into a pandas dataframe."""
    frame_index = 0

    # Because of the heavy weight of this dataset we decided to max it at 28 images.
    if latest_img > 28000:
        latest_img = 28000

    for i in range(latest_img + 1):
        try:
            emotion = np.load(path + "/annotations/"+ str(i)+"_exp.npy")
            img = mpimg.imread(path + "/images/"+str(i)+'.jpg')

            df.at[frame_index,'formatted_pixels'] = img
            df.at[frame_index,'target'] = emotions[int(emotion)]

            frame_index += 1

        # File not found is een onderdeel van IOE error oftewel een OS error 
        except OSError as e: 
            continue