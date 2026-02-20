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
		- When **error behaviour for $y$ is known and errors for $x$ are negligible**: use Bayesian methodology to write posterior pdf for model parameters as $$ p(\boldsymbol{\theta}| \{x_i, y_i\}, I) \propto p(\{x_i, y_i\} | \boldsymbol{\theta}, I) p(\boldsymbol{\theta}, I) $$where $I$ is the error behaviour; the *data likelihood* is given as: $$ p(y_i | x_i, \boldsymbol{\theta}, I) = e(y_i | y) $$where $y=f(x|\boldsymbol{\theta})$ is the model class, and $e(y_i | y)$ is the probability of observing $y_i$ given the true value/model prediction $y$
##### ***Data Sets Used in This Chapter***
- Data for relationships between redshifts of supernovae and their luminosity distances

### 8.2. Regression
- 326
##### ***Multivariate Regression***
##### ***Polynomial and Basis Function Regression***

### 8.3. Regularisation and Penalising the Likelihood
- 332
##### ***Ridge Regression***
##### ***LASSO Regression***
##### ***How Do We Choose the Regularisation Parameter $\lambda$***

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