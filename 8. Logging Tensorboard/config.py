#training hyperparameters

INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3


#dataset 
DATA_DIR = "dataset/"
NUM_WORKERS = 4


# compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16