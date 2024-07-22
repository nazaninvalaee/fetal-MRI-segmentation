import train
import evaluate
import predict

if __name__ == '__main__':
    print("Training the model...")
    train.main()
    print("Evaluating the model...")
    evaluate.main()
    print("Making predictions...")
    predict.main()
