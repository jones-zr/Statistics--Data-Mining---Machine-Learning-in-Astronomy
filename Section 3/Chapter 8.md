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
- 337

### 8.5. Kernel Regression
- 338

### 8.6. Locally Linear Regression
- 339

### 8.7. Nonlinear Regression
- 340

### 8.8. Uncertainties in the Data
- 342
##### ***Uncertainties in the Dependent and Independent Axes***

### 8.9. Regression That Is Robust to Outliers
- 343
##### ***Bayesian Outlier Methods***

### 8.10. Gaussian Process Regression
- 348

### 8.11. Overfitting, Underfitting, and Cross-Validation
- 352
##### ***Cross-Validation***
##### ***Learning Curves***
##### ***Other Cross-Validation Techniques***
##### ***Summary***

### 8.12. Which Regression Method Should I Use?
- 360
- 363
...

Next chapter: [[Chapter 9]]
New terminology: