*Regression and Model Fitting*
- Special case of general model fitting and procedure selection; the relation between a dependent variable $y$ and a set of independent variables $x$ that describes the **expectation value** of $y$ given $x$: $E[y|x]$
- The term *regression* is used as in "how a trend *regresses* toward a population mean"
- Regression techniques tend to make simplifying assumptions about data, uncertainties and models
### 8.1. Formulation of the Regression Problem
- Instead of finding an underlying pdf of a distribution, regression infers the **conditional expectation value**: $$y = f(x|\boldsymbol{\theta})$$where $y$ is a *scalar dependent variable*, $x$ is an *independent vector*, and $\boldsymbol{\theta}$ are parameters of the distribution
	- Here $x$ does not need to be a random variable, and for a given model of function $f$, there are $k$ model parameters $\theta_{p}$
- Observations $x_i$ and $y_i$ constrain model parameters $\boldsymbol{\theta}$; each point provides a joint constraint
	- Observations with no uncertainties provide constraints as lines; increasing number of points increases the constraints, and their intersection gives the best estimate of model parameters
	- Observations with uncertainties give constraints as a distribution rather than a line; best estimate of model parameters is then given by a posterior distribution
- General regression is computationally expensive
- **Classification axes** for regression problems:
	- *Linearity:* When a parametric model is linear in all model parameters, eg. $$ f(x|\theta) = \sum_{p=1}^{k} \theta_p g_p (x)$$regression becomes much simpler *linear regression*
	- *Problem complexity:* A large number of independent variables increases the complexity of the error covariance matrix
	- *Error behaviour:* Uncertainties of independent and dependent variables are the primary factor that determines choice of regression method
		- When **error behaviour for $y$ is known and errors for $x$ are negligible**: use Bayesian methodology to write *posterior pdf* for model parameters as $$ p(\boldsymbol{\theta}| \{x_i, y_i\}, I) \propto p(\{x_i, y_i\} | \boldsymbol{\theta}, I) p(\boldsymbol{\theta}, I) $$where $I$ is the error behaviour; the *data likelihood* is given as: $$ p(y_i | x_i, \boldsymbol{\theta}, I) = e(y_i | y) $$where $y=f(x|\boldsymbol{\theta})$ is the model class, and $e(y_i | y)$ is the probability of observing $y_i$ given the true value/model prediction $y$
##### ***Data Sets Used in This Chapter***
- Supernovae redshifts and their luminosity distances :D

### 8.2. Regression
- Linear model is the simplest case of regression: $$ y_i = \theta_0 + \theta_1 x_i + \epsilon_i $$where $\theta_0$ and $\theta_1$ are the coefficients that describe the regression function we are estimating (ie. slope and intercept of a line in this case), and $\epsilon_i$ is the additive noise term; uncertainties of independent variables $x_i$ are considered negligible, and dependent variables $y_i$ have known heteroscedastic uncertainties $\epsilon_i = \mathcal{N}(0,\sigma_i)$
- Therefore, *data likelihood* is written as: $$ p(\{y_i\} | \{x_i\}, \boldsymbol{\theta}, I) = \prod_{i=1}^{N} \frac{1}{\sqrt{2 \pi} \sigma_i} \exp \left( \frac{-(y_i - (\theta_0 + \theta_1 x_i))^{2}}{2 \sigma_{i}^{2}} \right)$$
- With no knowledge of the distribution of parameters (a *flat* or *uninformative* prior pdf, $p(\theta | I)$, the posterior will be directly proportional to the likelihood function (aka. the error function); the *logarithm of the posterior* gives the "classic definition" of regression in terms of the *log-likelihood*: $$ \ln (L) \equiv \ln ( p(\boldsymbol{\theta}| \{x_i, y_i\}, I)) \propto \sum_{i=1}^{N} \left( \frac{-(y_i - (\theta_0 + \theta_1 x_i))^{2}}{2 \sigma_{i}^{2}} \right) $$
- Maximising the log-likelihood as a function of model parameters is achieved by minimising the *sum of square errors*; comes from the assumption of Gaussian uncertainties for $y_i$
- Minimising $\ln(L)$ simplifies to: $$ \theta_1 = \frac{\sum_{i}^{N} x_i y_i - \overline{x} \overline{y}}{\sum_{i}^{N} (x_i - \overline{x})^2} \text{, and } \theta_0 = \overline{y} - \theta_1 \overline{x} $$where $\overline{x}$ and $\overline{y}$ are the mean values of $x$ and $y$
- For heteroscedastic errors, and for complex regression functions, its easier to use *matrix notation* to generalise regression problems, therefore define: $$ Y = M \boldsymbol{\theta} $$where $Y$ is an $N$-dimensional vector of values $y_i$, $$ Y = \begin{bmatrix} y_0 \cr y_1 \cr y_2 \cr . \cr y_{N-1} \end{bmatrix} $$the example straight-line regression function $\boldsymbol{\theta}$ is a 2D vector of regression coefficients, $$ \boldsymbol{\theta} = \begin{bmatrix} \theta_0 \cr \theta_1 \end{bmatrix} $$and design matrix $M$ is a $2 \times N$ matrix $$ M = \begin{bmatrix} 1 & x_0 \cr 1 & x_1 \cr 1 & x_2 \cr . & . \cr 1 & x_{N-1} \end{bmatrix} $$where the constant values capture the $\theta_0$ term in the regression
- For the heteroscedastic uncertainties, can define a *covariance matrix* $C$ as an $N \times N$ matrix: $$ C = \begin{bmatrix} \sigma_{0}^{2} & 0 & . & 0 \cr 0 & \sigma_{1}^{2} & . & 0 \cr . & . & . & . \cr 0 & 0 & . & \sigma_{N-1}^{2} \end{bmatrix} $$with the uncertainties $\sigma_i$ as the diagonals
- The *maximum likelihood solution* for this regression is $$ \boldsymbol{\theta} = (M^T C^{-1} M)^{-1} (M^T C^{-1} Y) $$which again minimises the sum of the square errors as above
- The uncertainties on the regression coefficients can be expressed as the symmetric matrix $$ \Sigma_\theta = \begin{bmatrix} \sigma_{\theta_0}^{2} & \sigma_{\theta_0 \theta_1} \cr \sigma_{\theta_0 \theta_1} & \sigma_{\theta_1}^{2} \end{bmatrix} = [M^T C^{-1} M]^{-1}$$
##### ***Multivariate Regression***
- Fitting a *hyperplane* rather than a straight line; extend the description of the regression function to multi-dimensions with $y = f(x | \boldsymbol{\theta})$ given by: $$ y_i = \theta_0 + \theta_1 x_{i1} + \theta_2 x_{i2} + ... + \theta_k x_{ik} + \epsilon_i $$where $x_{ik}$ is the $k$th component of the $i$th data entry within the multivariate data set; the *design matrix* $M$ is therefore: $$ M = \begin{bmatrix} 1 & x_{01} & x_{02} & . & x_{0k} \cr 1 & x_{11} & x_{12} & . & x_{1k} \cr . & . & . & . & . \cr 1 & x_{N1} & x_{N2} & . & x_{Nk} \end{bmatrix} $$
- Regression coefficients $\boldsymbol{\theta}$ and their uncertainties $\Sigma_{\boldsymbol{\theta}}$ equations are same as in the linear regression case
##### ***Polynomial and Basis Function Regression***
- A straight line can be interpreted as a first-order expansion of the regression function $y = f(x | \boldsymbol{\theta})$; therefore, $f(x | \boldsymbol{\theta})$ can be expressed as the sum of arbitrary functions as long as the model is linear in terms of the regression parameters $\boldsymbol{\theta}$ (if you're an insane maniac)
- Examples of general linear models: a Taylor expansion of $f(x)$ as a series of polynomials to solve for the amplitudes of the polynomials; a linear sum of Gaussians with fixed positions and variances to solve for the amplitudes of the Gaussians
- For *polynomial regression*, the regression function $f(x | \boldsymbol{\theta})$ becomes: $$  y_i = \theta_0 + \theta_1 x_{i} + \theta_2 x_{i}^{2} + \theta_3 x_{i}^{3} + ...  $$
- The *design matrix* $M$ becomes: $$ M = \begin{bmatrix} 1 & x_{0} & x_{0}^{2} & x_{0}^{3} \cr 1 & x_{1} & x_{1}^{2} & x_{1}^{3} \cr . & . & . & . \cr 1 & x_{N} & x_{N}^{2} & x_{N}^{3} \end{bmatrix} $$
- For data set with $k$ dimensions to which a $p$-dimensional polynomial is being fit to, the number of parameters in the model being fit is: $$ m = \frac{(p + k)!}{p! k!} $$including the intercept
	- Model's degrees of freedom is $\nu = N - m$; probability of the model is a $\chi^2$ distribution with $\nu$ degrees of freedom
	- Increased number of polynomial terms leads to overfitting
- Can generalise a polynomial model to a *basis function* representation by replacing rows in design matrix with a series of non/linear functions of the variables $x_i$
	- Problem stays linear despite arbitrary basis functions; only fitting the coefficients of the terms

### 8.3. Regularisation and Penalising the Likelihood
- **Regularisation (shrinkage):** limiting the complexity of the underlying regression model by applying a penalty to the likelihood function; trades increase in bias for a reduction in variance
- Eg. including a penalty/regularisation term on a minimisation: $$ (Y - M \boldsymbol{\theta})^T C^{-1} (Y - M \boldsymbol{\theta}) - \lambda |\boldsymbol{\theta}^T \boldsymbol{\theta}|^2 $$where $\lambda$ is the regularisation/*smoothing parameter*, and $|\boldsymbol{\theta}^T \boldsymbol{\theta}|^2$ is an example *penalty function*
	- This example penalises the size of the regression coefficients (**ridge regression**)
	- Solving for the parameters gives: $$ \boldsymbol{\theta} = (M^T C^{-1} M + \lambda I)^{-1} (M^T C^{-1} Y) $$
##### ***Ridge Regression***
- $L_2$ regularisation; penalty on the sum of the squares of the regression coefficients so that: $$ |\boldsymbol{\theta}|^2 < s $$where $s$ controls complexity of model similar to regularisation parameter $\lambda$
	- Suppressing large coefficients limits *variance* of the system but increases the bias of derived coefficients
	- Smaller $s$ values (higher $\lambda$ values) drive the regression coefficients towards zero
- Through SVD, *regularised regression coefficients* can be written as: $$ \boldsymbol{\theta} = V \Sigma' U^T Y $$where $\Sigma'$ is a diagonal matrix with elements $d_i / (d_i^2 + \lambda)$, with $d_i$ the eigenvalues of $MM^T$
	- As $\lambda$ increases, only the diagonal components with highest eigenvalues will contribute to the regression
- Goodness of fit for a ridge regression: $$ \hat{y} = M(M^TM + \lambda I)^{-1} M^T y $$and degrees of freedom: $$ \text{DOF} = \text{Trace}[M(M^TM + \lambda I)^{-1} M^T] = \sum_i \frac{d_i^2}{d_i^2 + \lambda} $$
##### ***LASSO Regression***
- $L_1$ regularisation; LASSO = "least absolute shrinkage and selection"; uses the $L_1$ norm to subset the variables within a model and apply shrinkage
	- Eg. LASSO penalises likelihood as: $$ (Y - M \boldsymbol{\theta})^T (Y - M \boldsymbol{\theta}) - \lambda |\boldsymbol{\theta}| $$where $|\boldsymbol{\theta}|$ penalises the absolute value of $\boldsymbol{\theta}$
- Equivalent to least-squares regression with penalty: $$ |\boldsymbol{\theta}| < s $$
- LASSO weights the regression coefficients and imposes *sparsity* on the regression model; as $\lambda$ increases, the size of the region encompassed with the constraint decreases
- Unlike ridge regression, LASSO has no *closed-form solution*
##### ***How Do We Choose the Regularisation Parameter $\lambda$***
- Estimating $\lambda$ involves minimising the cross-validation error; error defined by applying $k$-fold cross-validation techniques: $$ \text{Error}(\lambda) = k^{-1} \sum_k N_{k}^{-1} \sum_k^{N_k} \frac{[y_i - f(x_i | \boldsymbol{\theta})]^2}{\sigma_i^2} $$

### 8.4. Principal Component Regression
- For high-dimensional data or data sets with collinear variables; can define the *principal components* from the data covariance matrix $X^T X$ through EVD or SVD: $$ X^T X = V \Sigma V^T $$with $V^T$ the eigenvectors and $\Sigma$ the eigenvalues; projecting data matrix onto eigenvectors gives projected data points: $$ Z = X V^T $$and truncate to *exclude* components with small eigenvalues; *linear regression* can then be applied to transposed data: $$ Y = M_z \boldsymbol{\theta} + \epsilon $$with $M_z$ the design matrix for *projected components* $z_i$
	- Principal component analysis (PCA) plus truncation, and regression are two separate steps
- **PCR vs ridge regression**: number of model components in PCR is ordered by eigenvalues and is absolute, regression coefficients are weighted by 1 or 0; ridge regression weights coefficients continuously
	- PCR advantage: better for data with independent variables that are *collinear*
	- PCR disadvantage: truncation point it not set; eigenvalues do not always correlate with ability of principal components to predict dependent variables
- Most linear regressions implicitly do some kind of PCR to invert the matrix $M^T M$

### 8.5. Kernel Regression
- Defines a kernel, $K(x_i, x)$, *local to each data point*, with amplitude depending on the distance from the point to all other points in sample; *kernel is positive for all values* and approaches 0 asymptotically as distance approaches infinity; kernel influence is determined by width/*bandwidth* $h$
- *Nadaraya-Watson* estimation of the regression function is: $$ f(x|K) = \frac{\sum_{i=1}^{N} K \left( \frac{||x_i - x||}{h} \right) y_i}{\sum_{i=1}^{N} K \left( \frac{||x_i - x||}{h} \right)} $$which is taking the weighted average of the dependent variable $y$; gives higher weight $w_i(x)$ to points near $x$
- *Rule of thumb:* bandwidth is more important than exact shape of the kernel in kernel-based regression; estimation of the *optimal bandwidth* is done through cross-validation, and decreases with sample size as $N^{-1/5}$

### 8.6. Locally Linear Regression
- Solves a separate weighted least-squares problem at each point $x$, finding the $w(x)$ which minimises: $$ \sum_{i=1}^{N} K \left( \frac{||x - x_i||}{h} \right) (y_i - w(x) x_i)^2 $$
- Assume that regression function is approximated by a *Taylor series expansion* about any local point; truncating expansion at the first term (locally constant solution) gives *kernel regression*
- Function estimate of LLR is: $$ f(x|K) = \sum_{i=1}^{N} w_i(x) y_i $$
- A common form for $K(x)$ is the *tricubic kernel*: $$ K(x_i, x) = \left( 1 - \left( \frac{|x - x_i|}{h} \right)^3 \right)^3 $$
- **Local polynomial regression**: any polynomial order; going past linear increases variance without decreasing much bias, most boundary bias is captured in LLR
- **Variable-bandwidth kernels**: let the bandwidth for each point be inversely proportional to its $k$th nearest neighbour's distance

### 8.7. Nonlinear Regression
- Nonlinear optimisation problems require maximising the *posterior*, eg. thought MCMC
- Alternatively, use the *Levenberg-Marquardt (LM) algorithm* to optimise the maximum likelihood estimation; LM minimises the sum-of-squares error of a multivariate distribution: $$ \sum_i (y_i - f(x_i | \boldsymbol{\theta}_0) - J_i d\boldsymbol{\theta})^2 $$for the perturbation $d\boldsymbol{\theta}$, where $J$ is the Jacobian about the point
	- LM is iterative; at each iteration, searches for the step $d\boldsymbol{\theta}$ that minimises the sum-of-squares error, then updates regression model
	- Dampening parameters, $\lambda$, is adaptively varied each iteration, decreasing as the minimum is approached
	- At convergence, the regression parameter covariances are given by $[J^T \boldsymbol{\theta} J]^{-1}$
- Success of the LM algorithm often relies on the initial guess for the regression parameters being close to the maximum likelihood solution

### 8.8. Uncertainties in the Data
##### ***Uncertainties in the Dependent and Independent Axes***
- Both dependent and independent variables have measurement uncertainties
- Eg. for a linear model, the objective function is: $$ y_i^* = \theta_0 + \theta_1 x_i^*$$
	- Assuming $x$ and $y$ are noisy observations of $x^*$ and $y^*$: $$ x_i = x_i^* + \delta_i, \text{ and } y_i = y^* + \epsilon_i $$where $\delta$ and $\epsilon$ are centered normal distribution; solving for $y$ gives: $$ y = \theta_0 + \theta_1(x_i - \delta_i) + \epsilon_i $$
	- The uncertainty in $x$ is now *part of the regression equation* and scales with the regression coefficients; known as *total least squares* problem

### 8.9. Regression That Is Robust to Outliers
- Any regression or model fitting must be able to account for outliers from the fit
	- Standard *least-squares* regression that use an $L_2$ norm result in significant outliers, contributing as the square of systematic deviation
	- Known error distributions $e(y_i | y)$ can be included in the likelihood; unknown error distributions can be assumed or modelled as mixture models
	- An example of assuming an error distribution: adopting the $L_1$ norm, $\Sigma_i ||y_i - w_i x_i||$, which is *less sensitive* to outliers than the $L_2$ norm
		- Minimising the $L_1$ norm is essentially finding the *median*
- *Sigma clipping:* an iterative process that progressively prunes data points that are not well represented by the model
	- Approach formalised by *least-trimmed squares* by finding subset of $K$ points which minimises $\Sigma_i^K (y_i - \theta_i x_i)^2$; for large $N$, the number of combinations makes this computationally expensive
- *M (maximum-likelihood-type) estimators:* modifies the underlying likelihood estimator to be less sensitive that the classic $L_2$ norm
	- M estimator class includes maximum-likelihood approaches, like *least-squares* which minimises the sum of the squares of the residuals between data and model
	- Can replace least-squares with a different function, eg. Huber Loss Function
##### ***Bayesian Outlier Methods***
- *Main idea:* enhance a model so that it can naturally explain outliers, by:
	- Adding a background Gaussian component to data to create a *mixture model*; method is much less affected by outliers than a simple regression model
	- Identifying bad points and *rejecting outliers* through nuisance parameters; similar outlier sensitivity to mixture model method

### 8.10. Gaussian Process Regression
- Method is widely applicable to data that are not generated by a Gaussian process; creates flexible regression models that are more data driven
- *Gaussian process:* a collection of random variables in parameter space, any subset of which is defined by a joint Gaussian distribution; completely described by its *mean* and *covariance* function; defined for any number of dimensions for any positive covariance function
	- Eg. a 1D and squared-exponential covariance function: $$ \text{Cov}(x_1, x_2; h) = \exp \left ( \frac{-(x_1 - x_2)^2}{2h^2} \right) $$where $h$ is bandwidth; an infinite set of possible functions $f(x)$ can be drawn from this covariance space
	- Can *constrain* the Gaussian process by only selecting $f(x)$'s that pass through particular points
- For a *regression problem:* assume data are drawn from an underlying model $f(x)$, then estimate the mean value $\overline{f_j^*}$ and variance $\Sigma_{jk}^*$ for a new set of measurements $x_j^*$; in Bayesian terms, compute the posterior pdf: $$ p(f_i | \{x_i, y_i, \sigma_i \}, x_j^*) $$
	- This is equivalent to averaging over the entire set of possible functions $f(x)$ which pass through the constraints of the data; can change from an infinite function space to a finite covariance space by using a *"kernel trick"*
	- An important feature is that the process produces a *best-fit model*, the *uncertainty* at each point, and a *full covariance estimate* of the result at unknown points

### 8.11. Overfitting, Underfitting, and Cross-Validation
- The optimality of the regression is contingent on *correct model selection*; the frequentist tool of cross-validation is complementary to AIC and BIC
- Considering a simple model where $0 \leq x_i \leq 3$ and $y_i = x_i \sin(x_i) + \epsilon_i$, where noise $\epsilon_i \sim \mathcal{N}(0,0.1)$:
	- eg. simple linear fitting is found by minimising the mean square error: $$ \epsilon = \frac{1}{N} \sum_{i=1}^{N} (y_i - \theta_0 - \theta_1 x_i)^2 $$
	- This model is not a good fit, it is *biased* and *underfits* the data
	- A more complicated model with more free parameters should be a better fit, eg. a model with polynomial degree $d=19$ for 20 data points. This reduces the training error $\epsilon$ to 0, however this is also not a good fit, it has *high variance* and *overfits* the data
	- In this 2D case, increasing the degree of the polynomial leads to smaller training errors, but reflects overfitting the data rather than improving the underlying model; *cross-validation* can be used to quantitively evaluate the bias and variance of a regression model
##### ***Cross-Validation***
- Simplest method involves splitting training data into three: the training set (50% - 70%), the cross-validation set, and the test set
	- *Training set:* $d_1$; used to determine the parameters of a given model, then evaluate the training error $\epsilon_{tr}$ (same eq. as $\epsilon$ above)
	- *Cross-validation set:* $d_2$; used to evaluate the cross-validation error $\epsilon_{cv}$; as this set was not used to find parameters, $\epsilon_{cv}$ will be *large* for a *high-bias (overfit)* model, and better represents the true goodness of fit of the model
		- Model which *minimises* $\epsilon_{cv}$ is likely to be the best model
	- *Test set:* $d_0$; once model is determined, used to evaluate the test error, which gives estimate of reliability of the model for as unlabelled data set
		- Parameters (eg. $\theta$) are learned from the training set, *hyperparameters* (eg. $d$) are learned from the CV set; hyperparameters can also be overfit by the CV data set, so the test error is a better representation of the error expected for a new set of data
	- *Plotting* $\epsilon_{tr}$ and $\epsilon_{cv}$ by polynomial degree can visually show at which $d$ both are minimised
		- Small $d$: the model does not have enough complexity to describe the intrinsic features of the data
		- Large $d$: the model is overly complex and can match variations in the training set which do not reflect the underlying distribution
##### ***Learning Curves***
- Ways to *improve a model:*
	- Get more training data
	- Use a more/less complicated model (balance bias/variance)
	- Use more/less regularisation
	- Increase the number of features (eg. more observations of each object)
- With a fixed model, how can you *improve a data set*?
- A **learning curve** is the plot of the *truncated* training and CV errors as a function of number of training points, $n$, which is a truncated subset of the total training set, $n \leq N_{train}$
	- As $n$ increases, the *training error increases*; a model of a given complexity can better fit a small set of data than a larger one; always expect this to be true
	- As $n$ increases, the *CV error decreases*; a smaller $n$ leads to overfitting; overfitting is reduced with larger $n$
	- At all $n$, $\epsilon_{tr} \leq \epsilon_{cv}$; the model should better describe the data used to train it than the CV data
	- As $n$ becomes large, the training and CV curves will *converge* to the same value
		- When curves are separated by large amount: the model error is *dominated by variance* (overfitting), and additional training data will help
		- When curves converge: the model error is *dominated by bias* (underfitting), and additional data cannot improve the results *for this model*; requires a more sophisticated model or more features for each point
- Change both model and data to find best match between them
##### ***Other Cross-Validation Techniques***
- **Twofold Cross-Validation**: additionally training the model on the CV set and cross-validating on the training set
- **K-fold Cross-Validation**: splitting the data into $K+1$ sets ($d_0$ and $d_1 ... d_K$) and training $K$ different models; final training and CV errors are computed from the mean/median of the set of final results
- **Leave-One-Out Cross-Validation**: similar to K-fold CV, but each set has one data point each; can be useful on small data sets
- **Random Subset Cross-Validation**: training and CV sets are selected by randomly partitioning data, and repeating any number of times until the error statistics are well sampled; not every point is guaranteed to be used, so an outlier can affect results
##### ***Summary***
- "CV is one place where machine learning and data mining may be considered more of an art than a science."

### 8.12. Which Regression Method Should I Use?
- **Accuracy:**
	- PCR generally increases accuracy from basic linear regression by *treating collinearity* and *denoising* the data
	- Leap up when going from linear to *nonlinear* models
	- Leap up when going to *nonparametric* models, eg. kernel regression and Gaussian process regression
- **Interpretability:**
	- Linear models are easy to interpret
	- *Ridge/LASSO regression* increase interpretability by identifying the most important features
	- *Bayesian formulations* increase interpretability by making assumptions clear
	- Generalised linear models maintain the identity of the original columns; generalised nonlinear models become less interpretable
	- *Gaussian process regression* is "opaque"
- **Simplicity:**
	- Basic linear regression is simplest with no tunable parameters
	- Ridge/LASSO regression require only one tunable parameter
	- PCR similar with only one critical parameter
	- General local polynomial regression has an extra parameter in the polynomial order
- **Scalability:**
	- Linear regression is pretty good
	- LASSO becomes expensive as dimensionality increases
	- Kernel regressions can be sped up with fast tree-based algorithms
	- Gaussian process regression can also be sped up but is *very computationally expensive*


Next chapter: [[Chapter 9]]
New terminology: