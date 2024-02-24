import copy
import numpy as np
import matplotlib.pyplot as plt

m_train = 1000
height = 100
width = 100
rgb = 3

n_features = height * width * rgb

train_set_x_orig = np.random.randn(m_train, height, width, rgb)
train_set_y = [0] * m_train

m_test = 100
test_set_x_orig = np.random.randn(m_test, height, width, rgb)
test_set_y = [0] * m_train

# Reshape from [m, h, w, rgb] to [h*w*rgb, m]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalize dataset
train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = float(0)
    return w, b


def propagate(w, b, X, Y):
    # FORWARD PROPAGATION (FROM X TO COST)

    # compute activation
    A = sigmoid(np.dot(w.T, X) + b)

    # compute cost by using np.dot to perform multiplication.
    m = X.shape[1]
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # Redundant but safe
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations=2000,
    learning_rate=0.5,
    print_cost=False,
):
    # initialize parameters with zeros
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    # Gradient descent
    params, grads, costs = optimize(
        w,
        b,
        X_train,
        Y_train,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        print_cost=print_cost,
    )

    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print(
            "train accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
            )
        )
        print(
            "test accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
            )
        )

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }

    return d


w, b = initialize_with_zeros(n_features)

logistic_regression_model = model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True,
)

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(
        train_set_x,
        train_set_y,
        test_set_x,
        test_set_y,
        num_iterations=1500,
        learning_rate=lr,
        print_cost=False,
    )
    print("\n" + "-------------------------------------------------------" + "\n")

for lr in learning_rates:
    plt.plot(
        np.squeeze(models[str(lr)]["costs"]),
        label=str(models[str(lr)]["learning_rate"]),
    )

plt.ylabel("cost")
plt.xlabel("iterations (hundreds)")

legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()