from src import depthml as dml
from src.depthml.utils import get_xp
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
    dml.Input((1, 28, 28)),
    dml.Conv2d(filters=32, kernel_size=3, stride=1, padding=1),
    dml.LeakyReLU(),
    dml.Conv2d(filters=64, kernel_size=3, stride=2, padding=1),
    dml.LeakyReLU(),
    dml.Flatten(),
    dml.Linear(units=128),
    dml.LeakyReLU(),
    dml.Linear(units=10),
    device="gpu",
)


optim = dml.SGD(parameters=model.parameters(), learning_rate=0.01, momentum=0.9)
criterion = dml.SoftmaxCategoricalCrossentropy()

BATCH_SIZE = 32
BATCH_STEP = len(X_train) // BATCH_SIZE
EPOCHS = 5

for epoch in range(EPOCHS):
    start_time = time.time()
    total_acc = 0
    total_loss = 0

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    xp = get_xp(model.layers[1].w.data)

    for i in range(0, len(X_train), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]

        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        X_batch = dt.Tensor(X_batch, device=model.device)
        y_batch = dt.Tensor(y_batch, device=model.device)

        optim.zero_grad()

        y_pred = model(X_batch)
        loss = criterion(y_batch, y_pred)

        dt.differentiate(loss)
        dml.clip_grad_norm(optim.parameters, max_norm=1.0)
        optim.step()

        total_loss += loss.data

        predictions = xp.argmax(y_pred.data, axis=1)
        targets = xp.argmax(y_batch.data, axis=1)
        total_acc += xp.mean(predictions == targets)

        del y_pred

    avg_epoch_acc = total_acc.item() / BATCH_STEP
    avg_epoch_loss = total_loss.item() / BATCH_STEP

    optim.learning_rate *= 0.2

    print(
        f"[Epoch {epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_epoch_loss:.4f} "
        f"Accuracy: {avg_epoch_acc:.4f} "
        f"Time: {time.time() - start_time:.2f}s"
    )
