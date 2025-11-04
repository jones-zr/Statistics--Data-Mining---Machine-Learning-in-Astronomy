*About the Book and Supporting Material*

### 1.1. What Do Data Mining, Machine Learning, and Knowledge Discovery Mean?

##### ***Data Mining***
- A set of techniques for analysing and describing structured data; for example, finding patterns in large data sets
- Often interchangeable with the term *knowledge discovery*
- Results in the understanding of data set properties; it's not important to immediately contrast these data with a model
- High emphasis on **exploratory data analysis**: learning qualitative features of the data that were not previously known; **unsupervised learning**
##### ***Machine Learning***
- A set of techniques for interpreting data by comparing them to models for data behaviour
- Often known as inference techniques, data-based statistical inferences, or just **fitting**
- There can be more than one competing model and the data may indicate whether (at least) one of them can be rejected
- High emphasis on **prediction**: predicting one variable based on the other variables; **supervised learning**

### 1.2. What Is This Book About?
- Extracting knowledge (a quantitative summary of data behaviour) from data (results of measurements)
- **Probability density function** (pdf): $h(x)$, the probability that a value lies between $x$ and $x+dx$
	- The single most important problem in data mining is estimating the distribution $h(x)$ from which values of $x$ are drawn
- **Cumulative distribution function** (cdf): $H(x)$, the integral of the pdf
	- $H(x) = \int^{x}_{-\infty} h(x') dx'$
	- The inverse of the cdf is the **quantile function**
- **Population pdf**: the *true* pdf of the data sample, referred to with $h(x)$
- **Empirical pdf**: the data-derived estimate of the data sample, referred to with $f(x)$ (and it's cdf counterpart: $F(x)$)
	- $f(x)$ only tends to $h(x)$ if the data set is infinitely large. In cases that consider non-negligible measurement errors for $x$, $f(x)$ will never tend to $h(x)$ even with an infinite sample
	- $f(x)$ is a *model* of the true distribution $h(x)$; the functional form of $h(x)$ can only be guessed from constraining the model $f(x)$, which can be [[Parametric or Nonparametric]]
	- Once the functional form of the model is chosen, the best-fitting member of that model family, corresponding to the best setting of the model's parameters (eg. mean and standard deviation) must be chosen. Irrespective of the model's origin, a model can never be proved to be correct; it can only be tested against the data and sometimes rejected
		- Within a Bayesian framework, models cannot be rejected if they are the only one available; models can only be compared against each other and ranked on their success
- **Histogram**: the simplest nonparametric method to determine $f(x)$
- **Error distribution** or **uncertainty**: the probability of measuring value $x$ if the true value is $\mu$
	- $e(x) = p(x|\mu,I)$
	- where $I$ is all other information that specifies the details of the error distribution
	- The commonly used Gaussian (or normal) error distribution is:
	- $p(x|\mu,I) = \frac{1}{\sigma \sqrt{2\pi}} \exp(\frac{-(x-\mu)^{2}}{2 \sigma^{2}})$
- **Bias and scatter**: can include a bias $b$ in the error distribution, which is a systematic offset of all measurements from the true value $\mu$, and $\sigma$ controls the scatter of the trend
	- $(x-\mu)$ becomes $(x-b-\mu)$
- **Heteroscedastic errors**: when the error distribution is non-Gaussian or, even if it is, $\sigma$ is not the same for all measurements (ie. each measured $x_{i}$ has a different $\sigma_{i}$)
- **Homoscedastic errors**: where the error distribution is the same for each measurement
- **Classification** of data points $x$ with a 'class descriptor' can be important to data analysis
	- Can use a discrete tag (eg. star vs galaxy) or a continuous variable, often the probability of belonging to a class (eg. p(star) vs p(gal)), which reflects the current understanding of that object and its classification rather than it's true, unconfirmed nature

### 1.3. An Incomplete Survey of the Relevant Literature
Recommendations from the group:
- *Numerical Recipes: The Art of Scientific Computing*
- *Data Analysis: A Bayesian Tutorial*
- *Modern Statistical Methods for Astronomy With R Applications*
- *The Visual Display of Quantitative Information* (from Section 1.6.)

### 1.4. Introduction to the Python Language and the Git Code Management Tool
- **Python**: an open-source, object-oriented interpreted language
- **Core python packages for astro**: NumPy, SciPy, Matplotlib, Astropy, HealPy
- **Git**: a tool to manage, change, add, delete, merge files and file structures, and establish remote repositories

### 1.5. Description of Surveys and Data Sets Used in Examples
...

### 1.6. Plotting and Visualising the Data in This Book
...

### 1.7. How to Efficiently Use This Book
...


Next chapter: [[Chapter 2]]
Extra terminology: [[Parametric or Nonparametric]], [[Self-similar]]