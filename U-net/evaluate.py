# evaluate.py

from data_loader import get_data_splits
from tensorflow.keras.models import load_model
import config

def main():
    print("Loading data...")
    X_train, X_val, y_train, y_val = get_data_splits()
    
    print("Loading model...")
    model = load_model(config.MODEL_PATH)
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
