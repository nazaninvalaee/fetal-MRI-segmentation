import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from model import build_model
from data_loader import get_data_splits
import config
import sys
from tqdm import tqdm

class TrainingProgress(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_bar = tqdm(total=config.EPOCHS, position=0, desc='Epochs', file=sys.stdout)
        self.epoch_bar.n = epoch
        self.epoch_bar.refresh()
        print(f"\nStarting epoch {epoch + 1}/{config.EPOCHS}")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        print(f"Epoch {epoch + 1} complete. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    def on_train_end(self, logs=None):
        self.epoch_bar.close()
        print("Training complete.")

    def on_batch_end(self, batch, logs=None):
        print(f" - Batch {batch + 1} complete. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

def main():
    print("Building the model...")
    model = build_model(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES)
    
    print("Training the model...")
    X_train, X_val, y_train, y_val = get_data_splits()
    
    checkpoint_path = config.MODEL_PATH + '.keras'
    
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
    training_progress = TrainingProgress()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[checkpoint, training_progress]
    )
    
    model.save(config.MODEL_PATH)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
