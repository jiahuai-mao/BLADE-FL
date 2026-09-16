from __future__ import annotations

import math

import numpy as np


def network_dims(input_dim: int, hidden_dim, num_classes: int) -> list[int]:
    if isinstance(hidden_dim, (list, tuple)):
        hidden_dims = [int(item) for item in hidden_dim]
    else:
        hidden_dims = [int(hidden_dim)]
    dims = [int(input_dim), *hidden_dims, int(num_classes)]
    if any(item <= 0 for item in dims):
        raise ValueError("all network dimensions must be positive")
    return dims


def parameter_count(input_dim: int, hidden_dim, num_classes: int) -> int:
    dims = network_dims(input_dim, hidden_dim, num_classes)
    return sum(dims[index] * dims[index + 1] + dims[index + 1] for index in range(len(dims) - 1))


def initialize_model(input_dim: int, hidden_dim, num_classes: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    layers = []
    dims = network_dims(input_dim, hidden_dim, num_classes)
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        limit = np.sqrt(6.0 / float(fan_in + fan_out))
        weight = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
        bias = np.zeros(fan_out, dtype=np.float32)
        layers.append((weight, bias))
    return pack_layers(layers)


def pack_layers(layers) -> np.ndarray:
    parts = []
    for weight, bias in layers:
        parts.extend([np.asarray(weight).ravel(), np.asarray(bias).ravel()])
    return np.concatenate(parts).astype(np.float32, copy=False)


def unpack_layers(vector: np.ndarray, input_dim: int, hidden_dim, num_classes: int):
    vector = np.asarray(vector, dtype=np.float32)
    dims = network_dims(input_dim, hidden_dim, num_classes)
    expected = parameter_count(input_dim, hidden_dim, num_classes)
    if vector.size != expected:
        raise ValueError("model has %d parameters; expected %d" % (vector.size, expected))
    offset = 0
    layers = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        weight_size = fan_in * fan_out
        weight = vector[offset : offset + weight_size].reshape(fan_in, fan_out)
        offset += weight_size
        bias = vector[offset : offset + fan_out]
        offset += fan_out
        layers.append((weight, bias))
    return layers


def train_local(
    start_vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dim,
    num_classes: int,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
):
    vector = np.asarray(start_vector, dtype=np.float32).copy()
    velocity = np.zeros_like(vector)
    rng = np.random.RandomState(seed)
    losses = []
    batches = 0
    indices = np.arange(len(y), dtype=np.int64)
    for _ in range(local_epochs):
        rng.shuffle(indices)
        for begin in range(0, len(indices), batch_size):
            selected = indices[begin : begin + batch_size]
            if selected.size == 0:
                continue
            loss, gradient = loss_and_gradient(
                vector,
                x[selected],
                y[selected],
                input_dim,
                hidden_dim,
                num_classes,
                weight_decay,
            )
            velocity *= np.float32(momentum)
            velocity += gradient
            vector -= np.float32(learning_rate) * velocity
            losses.append(loss)
            batches += 1
    mean_loss = float(np.mean(losses)) if losses else None
    return vector.astype(np.float32, copy=False), mean_loss, batches


def loss_and_gradient(
    vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dim,
    num_classes: int,
    weight_decay: float = 0.0,
):
    layers = unpack_layers(vector, input_dim, hidden_dim, num_classes)
    activation = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    activations = [activation]
    preactivations = []
    for weight, bias in layers[:-1]:
        preactivation = np.dot(activation, weight) + bias
        activation = np.maximum(preactivation, np.float32(0.0))
        preactivations.append(preactivation)
        activations.append(activation)

    output_weight, output_bias = layers[-1]
    logits = np.dot(activation, output_weight) + output_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    batch = max(len(y), 1)
    chosen = np.maximum(probabilities[np.arange(len(y)), y], np.float32(1e-12))
    loss = -float(np.log(chosen).mean())
    if weight_decay:
        loss += 0.5 * float(weight_decay) * sum(float(np.sum(weight * weight)) for weight, _ in layers)

    delta = probabilities.copy()
    delta[np.arange(len(y)), y] -= np.float32(1.0)
    delta /= np.float32(batch)
    gradients = [None] * len(layers)
    for index in range(len(layers) - 1, -1, -1):
        weight, _ = layers[index]
        gradients[index] = (
            np.dot(activations[index].T, delta) + np.float32(weight_decay) * weight,
            delta.sum(axis=0),
        )
        if index > 0:
            delta = np.dot(delta, weight.T)
            delta[preactivations[index - 1] <= 0.0] = 0.0
    return loss, pack_layers(gradients)


def predict_logits(vector, x, input_dim: int, hidden_dim, num_classes: int):
    activation = np.asarray(x, dtype=np.float32)
    layers = unpack_layers(vector, input_dim, hidden_dim, num_classes)
    for weight, bias in layers[:-1]:
        activation = np.maximum(np.dot(activation, weight) + bias, np.float32(0.0))
    weight, bias = layers[-1]
    return np.dot(activation, weight) + bias


def evaluate(vector, x, y, input_dim: int, hidden_dim, num_classes: int, batch_size: int = 2048):
    predictions = []
    losses = []
    for begin in range(0, len(y), batch_size):
        batch_x = np.asarray(x[begin : begin + batch_size], dtype=np.float32)
        batch_y = np.asarray(y[begin : begin + batch_size], dtype=np.int64)
        logits = predict_logits(vector, batch_x, input_dim, hidden_dim, num_classes)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        losses.extend((-np.log(np.maximum(probabilities[np.arange(len(batch_y)), batch_y], 1e-12))).tolist())
        predictions.extend(np.argmax(probabilities, axis=1).tolist())
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(y, dtype=np.int64)
    accuracy = float(np.mean(pred == truth)) if len(truth) else 0.0
    f1s = []
    for label in range(num_classes):
        tp = int(np.sum((pred == label) & (truth == label)))
        fp = int(np.sum((pred == label) & (truth != label)))
        fn = int(np.sum((pred != label) & (truth == label)))
        denominator = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denominator) if denominator else 0.0)
    return float(np.mean(losses)) if losses else None, accuracy, float(np.mean(f1s))


def cosine_learning_rate(initial: float, minimum: float, update: int, total_updates: int) -> float:
    if total_updates <= 1 or update <= 1:
        return float(initial)
    progress = min(max((float(update) - 1.0) / (float(total_updates) - 1.0), 0.0), 1.0)
    return float(minimum + 0.5 * (initial - minimum) * (1.0 + math.cos(math.pi * progress)))


def weighted_average(vectors, weights):
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    result = np.zeros_like(np.asarray(vectors[0], dtype=np.float32))
    for vector, weight in zip(vectors, weights):
        result += np.asarray(vector, dtype=np.float32) * np.float32(float(weight) / total)
    return result
