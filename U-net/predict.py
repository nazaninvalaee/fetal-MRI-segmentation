# predict.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import config
from data_loader import get_data_splits

def main():
    print("Loading data...")
    _, X_val, _, y_val = get_data_splits()
    
    print("Loading model...")
    model = load_model(config.MODEL_PATH)
    
    print("Making predictions...")
    new_image = X_val[0].reshape(1, *config.INPUT_SIZE)
    prediction = model.predict(new_image)
    
    predicted_segmentation = np.argmax(prediction, axis=-1).reshape(config.INPUT_SIZE[:2])
    
    plt.imshow(predicted_segmentation, cmap='gray')
    plt.title('Predicted Segmentation')
    plt.show()

if __name__ == "__main__":
    main()
