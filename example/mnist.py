from src import depthml as dml
from .utils import load_mnist

import DepthTensor as dt
import numpy as np
import time

(X_train, y_train), (X_test, y_test) = load_mnist()


def one_hot(y, classes=10):
    return np.eye(classes)[y]


y_train = one_hot(y_train)
y_test = one_hot(y_test)

model = dml.Sequential(
    dml.Input((784,)),
    dml.Linear(units=784),
    dml.LeakyReLU(),
    dml.Linear(units=32),
    dml.LeakyReLU(),
    dml.Linear(units=10),
    device="gpu",
)

optim = dml.SGD(parameters=model.parameters(), learning_rate=0.1)
criterion = dml.SoftmaxCategoricalCrossentropy()

BATCH_SIZE = 64
BATCH_STEP = len(X_train) // BATCH_SIZE
EPOCHS = 5

for epoch in range(EPOCHS):
    start_time = time.time()
    avg_epoch_loss = 0
    avg_epoch_acc = 0

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    for i in range(0, len(X_train), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]

        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        X_batch = dt.Tensor(X_batch, device="gpu")
        y_batch = dt.Tensor(y_batch, device="gpu")

        optim.zero_grad()

        y_pred = model(X_batch)
        loss = criterion(y_batch, y_pred)

        dt.differentiate(loss)
        optim.step()
        avg_epoch_loss += loss.item()

        predictions = np.argmax(y_pred.data, axis=1)
        targets = np.argmax(y_batch.data, axis=1)
        avg_epoch_acc += np.mean(predictions == targets)

    avg_epoch_loss /= BATCH_STEP
    avg_epoch_acc /= BATCH_STEP
    print(
        f"[Epoch {epoch + 1}/{EPOCHS}] Loss: {avg_epoch_loss} Accuracy: {avg_epoch_acc} Time: {time.time() - start_time}"
    )
