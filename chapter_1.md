# Introduction
In a binary classification problem, the result is a discrete value output.

Logistic regression is a learning algorithm used in a supervised learning problem when the output 𝑦 are all either zero or one. 

The goal of logistic regression is to minimize the error between its predictions and training data.

![image](img\ch_1\img_1.jpg)

We say we have a collection of **m** objects **x**, called **X**:
- **X** is our dataset, dimension [**n, m**]. ~~It is posible to find implementation of X with dimension [m, n], but the prior implementation make this easier.~~
- **x** the input features vector [**n, 1**]

$$X = (x^{(1)}, x^{(2)}, ..., x^{(m-1)}, x^{(m)})$$

$$\text{where}\;x^{(i)} = (x^{(i)}_1, x^{(i)}_2, ..., x^{(i)}_{n-1}, x^{(i)}_n)^T$$

we can define:
- **m** as the number of examples in the dataset
- **n** as the number of features per example object

We have also a collection of labels for each element in X, the vector collecting those labels is called Y (dimension [**1, m**]) and is defined as follows:

$$Y = (y^{(1)}, y^{(2)}, ..., y^{(m-1)}, y^{(m)})$$

Using the logistic regression we want to predict y from x, for that we output an estimation based on probabilities:

$$ \text{Given}\, 𝑥^i,\; 𝑦̂^i = 𝑃(𝑦^i = 1|𝑥^i), \text{where}\; 0 ≤ 𝑦̂^i ≤ 1 $$

The Logistic regression estimation is represented as:

For one example $$x^{(i)}$$:

$$ 𝑦̂^i = a^i = 𝜎(z^i) = 𝜎(𝑤^{𝑇}𝑥^i + 𝑏) $$

$$𝑥^i ∈ R^{n_{x}}, \; 𝑦∈ [0,1], \; 𝑦̂ ∈ [0,1], \; 𝑤 ∈ R^{n_{x}}, \; 𝑏 ∈ ℝ; \;$$

$$\text{where} z^{(i)} = w^T x^{(i)} + b$$

The logistic regression uses te following parameters to estimate the input features vector:
- **w**: the weights, dimension [**n, 1**]
- **b**: the threshold, dimension [**1,1**]
- **𝜎**: the Sigmoid function

Where **z** is a linear function, but since we are looking for a probability constraint between [0, 1], the sigmoid function that is bounded between [0, 1] is used.

The sigmoid function is use just to translate the output number into probabilities.
 
$$\text{Sigmoid function}: 𝜎(𝑧)= \frac{1}{1+e^{-z}}$$

The sigmoid function has the following behavior:
- if 𝑧 is a large positive number, then 𝜎(𝑧) = 1
- if 𝑧 is small or large negative number, then 𝜎(𝑧) = 0
- if 𝑧 = 0, then 𝜎(𝑧) = 0.5

> Note: from Sigmoid to ReLU
>
> Switching from the sigmoid to the ReLU activation function in the gradient descent optimization algorithm significantly expedited convergence. This improvement stems from the fact that the sigmoid function yields extremely small values for inputs less than zero, which approach zero but are not precisely zero. Consequently, learning with these minuscule numbers was notably sluggish.

**L** called the **loss function** is a function will need to define to measure how good our output y^ is when the true label is y. **L** is defined with respect to a single training example: it measures how well you're doing on a single training example.

The **Loss function** is computed as follows:

$$\mathcal{L}(a^{(i)}, y^{(i)}) = - y^{(i)} \log(a^{(i)}) - (1-y^{(i)} ) \log(1-a^{(i)})$$

>The Loss function could have been computed as well as
>
> $$\mathcal{L}(a^{(i)}, y^{(i)}) = \frac{1}{2}( y^{(i)} - y^{(i)} )^2$$
>
> the only problem is that this function is non convex and has lots of different local optimal. We are looking for a function that could explain the probability of having 1 or 0 and the former function explains this probability. This note can be expanded, see biblio.

The **cost function** measures how are you doing on the entire training set. So in training your logistic regression model, we're going to try to find parameters **w** and **b** that minimize the overall cost function **J**.

The **cost function** is then computed by summing over all training examples:
 
$$J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$$

## Build a logistic regression classifier
You've implemented several functions that:

- Initialize **w** and **b**
- Create a for loop over each sample to Optimize the loss iteratively to learn parameters (w,b):
 - Calculate current loss: forward propagation
 - Calculate current gradient: backward propagation
 - Update parameters: gradient descent

- Use the learned **w** and **b** to predict the labels for a given set of examples

### Initializing parameters
What we want to do is to find the value of **w** and **b** that corresponds to the minimum of the cost function **J**; and because this function is convex, no matter where you initialize, you should get to the same point.

Initialize **w** as a vector of zeros and **b**=0.

### Parameters learning
You can use the gradient descent algorithm to train or to learn the parameters **w** and **b** on your training set.

### Forward propagation
Compute the cost function **J**:

$$J = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$$

### Backwards propagation
Find the gradient **dw** and **db**: the goal is to learn **w** and **b** by minimizing the cost function **J**.

$$A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$$

$$dw =\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$

$$ db=\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$

**w** has dimension [**n, 1**]; X has dimension [**n, m**]; A has dimension [**1, m**].

In this example we used **vectorization**. The python numpy library exploit parallelization to enhance the computational time. In fact, both GPU and CPU have parallelization instructions: they're sometimes called SIMD instructions (Single Instruction Multiple Data.

### Optimization:
Update the parameters using gradient descent method

$$w = w - \alpha \text{ } dw\;$$

$$b = b - \alpha \text{ } db\;$$

where $$\alpha$$ is the learning rate

### Prediction
To perform the prediction two steps are necessary.

- Compute $$\hat{Y} = A = \sigma(w^T X + b)$$

- then perform the binarization: Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5)
