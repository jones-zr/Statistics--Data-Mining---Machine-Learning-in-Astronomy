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
- 366

### 9.5. Discriminative Classification
- 367
##### ***Logistic Regression***

### 9.6. Support Vector Machine
- 370

### 9.7. Decision Trees
- 373
##### ***Defining the Split Criterion***
##### ***Building the Tree***
##### ***Bagging and Random Forests***
##### ***Boosting Classification***

### 9.8. Deep Learning and Neural Networks
- 381
##### ***Neural Networks***
##### ***Training the Network***
##### ***How Many Layers and How Many Neurons?***
##### ***Convolutional Networks***
##### ***Autoencoders***

### 9.9. Evaluating Classifiers: ROC Curves
- 391

### 9.10. Which Classifier Should I Use?
- 393

...