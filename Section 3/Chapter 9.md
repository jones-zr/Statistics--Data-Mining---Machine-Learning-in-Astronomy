*Classification*
- [[Chapter 6#6.4. Finding Clusters in Data|Chp 6.4]] describes *unsupervised classification* to find inherent clustering within data properties
- **Supervised classification:** utilising labels to develop a relationship between labels and properties
- Simple classification schemes with few labels have more impact than those with many labels

### 9.1. Data Sets Used in This Chapter
- RR Lyrae, quasars, and photometric redshifts (Oh My!)

### 9.2. Assigning Categories: Classification
- Supervised classification takes a set of features and relates them to predefined sets of classes; goal is to *characterise the relation between the features in the data and their classes*, then apply these classifications to a larger set of unlabelled data
- **Generative classification:** classification done through density estimation; has advantage of high interpretability
- **Discriminative classification:** classification based on finding the decision boundary that separates classes; better in higher-dimensional problems
##### ***Classification Loss***
- **Zero-one loss:** most common loss/cost function; assign a value of 1 for a misclassification and 0 for a correct classification: $$ L(y, \hat{y}) = \delta (y \neq \hat{y}) $$
- Classification *risk* of a model (the *misclassification* or *error rate*) is the expectation value of the loss: $$ \mathbb{E} [L(y, \hat{y})] = p(y \neq \hat{y}) $$
- *Completeness:* fraction of total detections identified by the classifier $$ \text{completeness} = \frac{\text{true +ve}}{\text{(true +ve) + (false -ve)}} $$
- *Contamination:* fraction of detected objects which are misclassified $$ \text{contamination} = \frac{\text{false +ve}}{\text{(true +ve) + (false +ve)}} $$
- *Efficiency:* (in astronomy) $1 - \text{contamination}$

### 9.3. Generative Classification
- For data $\{ \mathbf{x} \}$ of $N$ points in $D$ dimensions, with a set of discrete labels $\{y\}$ from $K$ classes with values $y_k$, *Bayes' theorem* describes the relation between labels and features as: $$ p(y_k | \mathbf{x}_i) = \frac{p(\mathbf{x}_i | y_k) p(y_k)}{\Sigma_i p(\mathbf{x}_i | y_k) p(y_k)} $$
##### ***General Concepts of Generative Classification***
- **Discriminant Function:**
	- Classification is analogous to regression, where $y$ is categorical (eg. $y=\{0,1\}$); the expectation value from regression is now the *discriminant function*: $$ g(\mathbf{x}) = f(y|\mathbf{x}) = \int y p(y|\mathbf{x}) dy $$
	- Applying Bayes' rule to this gives: $$ g(\mathbf{x}) = \frac{\pi_1 p_1(\mathbf{x})}{\pi_1 p_1(\mathbf{x}) + \pi_0 p_0(\mathbf{x})} $$where $\pi \equiv p(y=y_k)$
- **Bayes Classifier:**
	- Making the discriminant function yield a *binary prediction* gives a Bayes classifier: $$ \hat{y} = \begin{cases} 1 & \text{if $g(\mathbf{x}) > 1/2$,} \\ 0 & \text{otherwise,} \end{cases} $$ $$ \hat{y} = \begin{cases} 1 & \text{if $p(y=1|\mathbf{x}) > p(y=0|\mathbf{x})$,} \\ 0 & \text{otherwise,} \end{cases} $$ $$ \hat{y} = \begin{cases} 1 & \text{if $\pi_1 p_1(\mathbf{x}) > \pi_0 p_0(\mathbf{x})$,} \\ 0 & \text{otherwise.} \end{cases} $$
	- This is a *template* as difference models for the $p_k$ and $\pi$ quantities can be used; "Bayesian" only in the utilisation of Bayes' rule, not Bayesian inference
- **Decision Boundary:**
	- Boundary between two classes is the set of $x$ values at which each class is *equally likely*, eg. $g_1(\mathbf{x}) = g_2(\mathbf{x})$, or $\pi_1 p_1(\mathbf{x}) = \pi_2 p_2(\mathbf{x})$
##### ***Naive Bayes***
- Bayes classifier can be difficult to compute on a high-dimensional $\{ \mathbf{x} \}$; reduce complexity by the assumption that all the attributes measured are *conditionally independent*: $$ p(x^0, x^1, x^2, ..., x^N|y_k) = \prod_i p(x^i|y_k) $$
- A general prescription for the naive Bayes' estimator: $$ \hat{y} = \arg\max_{y_k} \frac{\prod_i p(x^i|y_k) p(y_k)}{\sum_j \prod_i p(x^i|y_j) p(y_j)} = \arg \max_{y_k} \frac{\prod_i p_k(x^i) \pi_k}{\sum_j \prod_i p_j(x^i) \pi_j} $$
	- After applying Bayes' rule, conditional independence, then maximising over $y_k$, to general generative classification equation above
	- Once models for $p_k(x^i)$ and $\pi_k$ are known, the estimator $\hat{y}$ can be computed
##### ***Gaussian Naive Bayes and Gaussian Bayes Classifiers***
- **Gaussian naive Bayes:** each probability $p_k(x^i)$ is modelled as a 1D normal distribution, with determined means $\mu_k^i$ and widths $\sigma_k^i$
	- Estimator $\hat{y}$ can be written: $$ \hat{y} = \arg \max_{y_k} \left[ \ln \pi_k - \frac{1}{2} \sum_{i=1}^{N} \left( \ln[2 \pi (\sigma_k^i)^2] + \frac{(x^i - \mu_k^i)^2}{(\sigma_k^i)^2} \right) \right] $$with the log of the Bayes criterion, and omitted normalisation constant
	- Assumes the multivariate distribution $p(\mathbf{x}|y_k)$ can be modelled using as *axis-aligned multivariate Gaussian distribution*
- *Figure 9.2* shows easy classification; in real data, distributions overlap and categories have imbalanced numbers
- **Gaussian Bayes classifier:** allowing the Gaussian probability model for each class to have arbitrary correlations (ie. relaxing assumption of conditional independence / *allowing covariance*)
	- Estimator $\hat{y}$ can be written: $$ \hat{y} = \begin{cases} 1 & \text{if $m_1^2 < m_0^2 + 2\log(\frac{\pi_1}{\pi_0}) + (\frac{|\Sigma_1|}{|\Sigma_0|})$,} \\ 0 & \text{otherwise,} \end{cases} $$where $\Sigma_k$ is a $D \times D$ symmetric covariance matrix with $\text{det}(\Sigma_k) \equiv |\Sigma_k|$, and $m_k^2 = (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)$ is the *Mahalanobis distance*
##### ***Linear Discriminant Analysis and Relatives***
- *LDA* assumes that the distributions of $p_k(\mathbf{x})$ have *identical covariances* for all $K$ classes; all classes are a set of shifted Gaussians
- Optimal classifier: $$ g_k(\mathbf{x}) = \mathbf{x}^T \Sigma^{-1} \mu_{\mathbf{k}} - \frac{1}{2} \mu_{\mathbf{k}}^T \Sigma^{-1} \mu_{\mathbf{k}} + \log \pi_k $$where $\mu_{\mathbf{k}}$ is the mean of class $k$, and $\Sigma$ the covariance of the Gaussians
	- The Bayes classifier is *linear* with respect to $\mathbf{x}$
- Discriminant boundary between classes is line that *minimises* overlap between Gaussians
	- Relaxing assumption that $\Sigma$ is constant, the discriminant function $g(\mathbf{x})$ becomes *quadratic* in $x$; *quadratic discriminant analysis (QDA)*
- **Fisher's linear discriminant (FLD):** all priors are set equal without covariances being equal; projects all data onto a single line; classification boundary found by minimising the loss over all possible lines
##### ***More Flexible Density Models: Mixtures and Kernel Density Estimates***
- **Gaussian mixture model (GMM) Bayes classifier:** using GMMs to represent each $p_k(x)$ to more closely model the true distributions, as in [[Chapter 6#6.3. Parametric Density Estimation|Chp 6.3]]
	- Gaussian components $K$ must be chosen for each class independently, and cost of model fitting for each value of $K$
	- Classes with $K=1$ resemble naive Bayes classifier, but here Gaussians are allowed to have arbitrary covariances between dimensions, resulting in better fits
- **Kernel discrimination analysis:** nonparametric Bayes classifier that models each class with a *kernel density estimate*
	- Method takes GMM to natural limit; unlike GMM, no need to optimise over the locations of the mixture components; locations are the training points themselves
	- Only one variable to optimise, the bandwidth; allows for parameters to maximise classification performance rather than density estimation performance
	- Downside: high computation cost of evaluating kernel density estimates

### 9.4. $K$-Nearest-Neighbour Classifier
- *Using the class label of the nearest point*; an "approximate" kernel discriminant analysis with a variable bandwidth
- Simplest classifier assumes nothing about the form of the conditional density distribution (ie. is completely *nonparametric*); resulting decision boundary is a *Voronoi tessellation* of the attribute space
- Number of neighbours $K$ is used as a smoothing parameter; increasing $K$ increases *variance* but increases *bias*
- Weights can be assigned to each point's "vote" by weighing by the distance to the nearest point; classifier is directly related to *kernel regression*
- Euclidean distance used for distance metric
- Works best with large data samples; computational time to search for neighbours can be expensive

### 9.5. Discriminative Classification
- Directly modelling the decision boundary between two or more classes of source, as above
##### ***Logistic Regression***
- Can be in the form of two (binomial) or more (multinomial) classes; name from the *logistic function*, $e^x / (1+e^x)$
- When $y$ is binary, it can be modelled as a *Bernoulli distribution* with conditional likelihood function of: $$ L(\beta) = \prod_{i=1}^N p_i(\beta)^{y_i} (1 - p_i(\beta))^{1-y_i} $$
- The *model* is by assumption: $$ \log \left( \frac{p(y=1|x)}{p(y=0|x)} \right) = \beta_0 + \beta^T x $$
	- Parameters are chosen to effectively minimise classification error rather than density estimation error

### 9.6. Support Vector Machine
- Assuming the classes are *linearly separable* as $y=\{-1,1\}$; consider a hyperplane that maximises distance of the closest point from either class; this distance is the *margin*, points on the margin are *support vectors*
- Hyperplane maximising the margin is: $$ \max_{\beta_0, \beta}(m) \text{ subject to } \frac{1}{||\beta||} y_i (\beta_0 + \beta^T x_i) \geq m \text{ $\forall$ $i$} $$
- Optimising this problem is equivalent to minimising: $$ \frac{1}{2} ||\beta|| \text{ subject to } y_i(\beta_0 + \beta^T x_i) \geq 1 \text{ $\forall$ $i$} $$
	- A *quadratic programming* problem with many known solutions
- The discriminant function for the hyperplane can be written as: $$ g(x) = \beta_0 + \sum_{i=1}^N \alpha_i y_i \left< x, x_i \right> $$where $\alpha$ is a vector of weights with $\alpha_i>0$ and $\sum_i \alpha_i y_i = 0$
- Major limitation: limited to *linear decision boundaries*; becomes nonlinear through *kernalisation*
	- Replacing each occurrence of $\left< x_i, x_{i'} \right>$ with a kernel function $K(x_i, x_{i'})$

### 9.7. Decision Trees
- Hierarchical decision boundaries; each branch is subdivided into two child *nodes* based on a *predefined* decision boundary; boundaries are *axis aligned*; splitting is recursive up to a predefined stopping criterion
- Full decision tree is a function of the number of features used; depth of tree affects precision and accuracy of classification; last nodes are 'terminal nodes' or '*leaf nodes*'; simple, easy to visualise and interpret classifiers
##### ***Defining the Split Criterion***
- **Entropy:** $$ E(x) = -\sum_i p_i(x) \ln(p_i(x)) $$
- **Information gain:** the reduction in entropy due to partitioning the data; for a binary split: $$ IG(x|x_i) = E(x) - \sum_{i=0}^1 \frac{N_i}{N} E(x_i) $$where $N_i$ is the number of points in $x_i$, and $E(x_i)$ is the entropy associated with that class
- Optimal decision boundaries are computationally expensive to find; each feature is considered one by one, and the feature with the largest IG is split; the split value is similarly found: sorting data on feature $i$ and maximising the IM for a given splitting point $s$
- Common loss functions:
	- The **Gini coefficient** estimates the probability that a source would be incorrectly classified if it was chosen at random from a data set and the label selected randomly based on the distribution of classifications; for a $k$-class sample: $$ G = \sum_i^k p_i(1-p_i) $$
	- The **misclassification error** is the fractional probability that a point selected at random will be misclassified: $$ MC = 1 - \max_i(p_i) $$
##### ***Building the Tree***
- Common criterion for stopping recursive splitting: a node contains *only one class* of object; a split *does not improve IG* or reduce misclassifications; or number of points per node reaches a *predefined value*
- Complexity defined by number of levels/depth of the tree; increase depth will *decrease training error* but eventually lead to overfitting; *cross-validation* can be used to optimise depth
##### ***Bagging and Random Forests***
- Both examples of *ensemble learning*: idea of combining outputs of multiple models through voting or averaging
- **Bagging:** averages the predictive results of a series of bootstrap samples from training set; applicable to regression and nonlinear model fitting or classification techniques
	- Bagging estimator: $$ f(x) = \frac{1}{N} \sum_i^K f_i(x) $$where $N$ is the sample of points in the training set, there are $K$ equally sized bootstrap samples from which to estimate the function $f_i(x)$
- **Random forests:** expand bagging by generating a *set of decision trees* from these bootstrap samples; classification is from averaging the classifications of each individual decision tree
	- Addresses limitations of decision trees: overfitting of the data if the trees are deep, and that axis-aligned partitioning doesn't accurately reflect the correlated and/or nonlinear decision boundaries within data
	- Define *n*, number of trees to generate, and *m*, number of attributes to consider splitting at each level; at each node, a set of $m$ variables are randomly selected and the split criterion evaluated from them; keeping $m$ small reduces complexity and concerns of overfitting
	- Classification derived from the *mean or mode* of the results of all trees
	- CV can also optimise $m$ and $n$; $m$ is often chosen to be $\sim \sqrt{K}$, with $K$ the number of total attributes in the sample
##### ***Boosting Classification***
- Idea that combining many weak classifiers can result in improved classification; creates a *new model* to correct the errors of the *ensemble* so far; *re-weighting* the data based on previous incorrect classification
- After all iterations, final classification is given by the weighted votes of each classifier
- Limitations: computation time for large sets due to reliance on chain of dependent classifiers

### 9.8. Deep Learning and Neural Networks
##### ***Neural Networks***
- *Neuron:* core computational unit that takes a series of inputs from branched extensions, operates on them (applies a nonlinear function), and generates an output that is transmitted to one or more neurons; *networks:* connections of layers of neurons to one another
- Output of any neuron $j$ is: $$ a_j = f \left( \sum_i w_{ij} x_i + b_i \right) $$where $x_i$ is a set on inputs, $w_{ij}$ is a weighted value, and $b_j$ is a bias term
- Neurons between the input/output layers are *hidden layers*; *fully connected layers* have neurons connected to all neurons in the subsequent layer; *feed-forward networks* have outputs only connected to subsequent layers
- Output of the final neuron in the output layer: $$ o_k = f \left( \sum_i w_{jk} a_j + b_k \right) $$
##### ***Training the Network***
- Concept: given a labelled set of data and a loss function, optimise the weights and biases within the network by minimising the loss
- **Backpropagation:** backwards propagation of errors; a type of automated differentiation; uses the *chain rule* to decompose the derivative of a complex function into a sequence of operations, then apply standard gradient descent approaches for learning
- Rewriting input into a neuron as $z_k = \sum_j w_{jk} a_j$, the derivative of the loss is a function of its input weights: $$ \frac{\partial L}{\partial w_{jk}} = (y - a_k) a_k (1 - a_k) a_j $$which expresses the *gradient of the loss* in terms of the activation values from the previous layer
- Derivative of the loss w.r.t the inputs to the hidden layer: $$ \frac{\partial L}{\partial w_{ij}} = \left[ \sum_k \delta_k w_{jk} \right] a_j (1 - a_j) a_i $$
- Optimisation of network starts with random weights and biases; propagates these forward to calculate loss, then estimates the derivatives via backpropagation; weights are *updated* as: $$ w_{ij}' = w_{ij} - \alpha \frac{\partial L}{\partial w_{ij}} $$where $\alpha$ is the *learning rate*; iterative process continues until convergence of the loss
##### ***How Many Layers and How Many Neurons?***
- **Network architecture:** how to configure the network
- Too few neurons leads to underfitting; too many is overfitting
- Recommendation is to start with one layer; few problems require more than two; CV can be used to determine the need for more layers
- Rules of thumb:
	- No. of neurons between no. of input and output nodes
	- No. of neurons ~ no. of outputs + 2/3 the number of input nodes
	- no. of neurons in the hidden layer $< 2 \times$ size of input layer
##### ***Convolutional Networks***
- CNNs; designed to work with *images* and *time series* data; requires that the output of each pixel be connected to every neutron in the first layer
- Reduce complexity by requiring neurons only respond to inputs from a subset of an image (the *receptive field*)
- Four principal components to a CNN: a convolution layer; a nonlinear activation layer; a pooling or downsampling operation; and a fully connected layer for classification
- "Convolution" refers to the convolution of the input data $I(x,y)$ with a kernel $K(x,y)$ to produce a *feature map* $F(x,y)$: $$ F(x,y) = K(x,y) * I(x,y) = \sum_{x_0} \sum_{y_0} I(x - x_0, y - y_0) K(x_0, y_0) $$
	- Kernel only responds to pixels within its receptive field (ie. size of the kernel); kernels described by a *depth* (no. of kernels $K$) and a *stride* (how many pixels a kernel shifts at each step; typically 1)
- Nonlinear *activation function* is applied to the pixels of the feature map; *pooling* summarises the feature map values within a region of interest with either mean or max pixel value (max pooling); network training is then done with backpropagation
- Final layer is the classification layer; maps the outputs to a set of labels
##### ***Autoencoders***
- Variants of NN that learn the encoding or structure within data; kind of two networks: *an encoder* which learns a representation of the data, and *a decoder* that reconstructs the input data from the representation
- Traditional autoencoders generally used only for denoising data and not generating new data samples

### 9.9. Evaluating Classifiers: ROC Curves
- Evaluation heavily relies on preference for completeness vs contamination
- **Receiver operating characteristic (ROC) curves:** show the true +ve rate as a function of the false +ve rate as the discriminant function is varied (eg. Figure 9.23); best curves are closest to upper left corner
	- When sources are rare, more informative to plot *efficiency vs completeness*

### 9.10. Which Classifier Should I Use?
- **Accuracy:**
	- Not clear in advance of testing ("no free lunch")
	- In general, NN are more flexible and excel at image/time series classification
	- An ensemble of models will generally yield the highest accuracy
- **Interpretability:**
	- Parametric methods the most interpretable; nonparametric methods often too large to be interpretable
	- Decision trees are highly intuative
- **Scalability:**
	- Naive Bayes and variants are the easiest
	- Kernalised support vector machines are the worst
- **Simplicity:**
	- Naive Bayes are the simplest in both implementation and learning
	- All others are mostly good

### Table 9.1

| Method                              | Accuracy | Interpretability | Simplicity | Speed |
| ----------------------------------- | -------- | ---------------- | ---------- | ----- |
| Deep neural networks                | H        | M                | M          | M     |
| Naive Bayes classifier              | L        | H                | H          | H     |
| Mixture Bayes classifier            | M        | H                | H          | M     |
| **Kernel discriminant analysis**    | **H**    | **H**            | **H**      | **M** |
| Neural networks                     | H        | L                | L          | M     |
| Logistic regression                 | L        | M                | H          | M     |
| Support vector machines: linear     | L        | M                | M          | M     |
| Support vector machines: kernelised | H        | L                | L          | L     |
| **$K$-nearest-neightbour**          | **H**    | **H**            | **H**      | **M** |
| Decision trees                      | M        | H                | H          | M     |
| Random forests                      | H        | M                | M          | M     |
| Boosting                            | H        | L                | L          | L     |


Next chapter:
New terminology:

...
Random variable
L norms