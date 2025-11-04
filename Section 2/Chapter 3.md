*Probability and Statistical Distributions*
### 3.1. Brief Overview of Probability and Random Variables
##### ***Probability Axioms***
- Probability of A, $p(A)$, must satisfy three **Kolmogorov axioms**:
	- $p(A) \geq 0$ for each $A$
	- $p(\Omega) = 1$, where $\Omega$ is a set of all possible outcomes $A$
	- If $A_1, A_2, ...$ are disjoint events, then $p(\bigcup^{\inf}_{i=1} A_i) = \sum^{\inf}_{i=1} p(A_i)$, where $\bigcup$ stands for "union"
- Probability of the **union** of events A and B (**sum rule**): $$ p(A \bigcup B) = p(A) + p(B) - p(A \bigcap B) $$ where $\bigcap$ stands for "intersection"
- Complementary events: $$ p(A) + p(\overline{A}) = 1 $$
- Intersection of events: $$ p(A \bigcap B) = p(A|B)p(B) = p(B|A)p(A) $$
- **Law of total probability**: $$ p(A) = \sum_{i} p(A \bigcap B_i) = \sum_{i} p(A|B_i)p(B_i) $$ if events $B_i, i = 1, ..., N$ are disjoint
- **Classical statistical inference** is concerned with $p(A)$, the long-term outcome or frequency with which $A$ would occur; **Bayesian inference** is concerned with $p(A|B)$, the plausibility of proposition $A$ conditional on the truth of $B$
##### ***Random (Stochastic) Variables***
- A variable whose value results from the measurement of a quantity that is subject to random variations
- **Discrete** random variables form a countable set; **continuous** random variables usually map onto the real number set
- **Independent identically distributed (iid)** random variables are drawn from the same distribution and are independent, ie., knowing variable $x$ tells nothing about variable $y$: $$ p(x,y) = p(x)p(y) $$
##### ***Conditional Probability and Bayes' Rule***
- When two **continuous** random variables are **not independent**: $$ p(x,y) = p(x|y)p(y) = p(y|x)p(x) $$
- **Marginal probability:** $$ p(x) = \int p(x|y) dy $$
- **Continuous** law of total probability: $$ p(x) = \int p(x|y)p(y) dy $$
- All combine to give **Bayes' rule:** $$ p(y|x) = \frac{p(x|y)p(y)}{p(x)} = \frac{p(x|y)p(y)}{\int p(x|y)p(y) dy} $$ where:
	- $p(y|x)$ is the **posterior**
	- $p(x|y)$ is the **data**
	- $p(y)$ is the **prior**
	- $p(x)$ is the **normalisation**
##### ***Transformation of Random Variables***
- Any function of a random variable is itself a random variable
- eg. for probability density distribution $p(x)$, where $x$ is a random variable, and $y = \Phi (x)$: $$ p(y) = p[\Phi^{-1}(y)] \left|\frac{d \Phi^{-1}(y)}{dy} \right| = p(x) \left|\frac{dx}{dy} \right| $$
- If $y = \Phi(x) = \exp(x)$, then $x=\Phi^{-1}(y) = \ln(y)$, and if $p(x) = 1$ for $0 \leq x \leq 1$ and $0$ otherwise (a uniform distribution), then $p(y) = 1/y$ for $1 \leq y \leq e$

### 3.2. Descriptive Statistics
- Any distribution function $h(x)$ can be characterised by it's **location** parameters, **scale/width** parameters, and **shape** parameters; when these parameters are based on the true, underlying distribution $h(x)$, they are **population** statistics; when they are based on a finite-size data set $f(x)$, they are estimated **sample** statistics
##### ***Definitions of Descriptive Statistics***
- **Arithmetic mean/expectation value:** $$ \mu = E(x) = \int^{\infty}_{-\infty} xh(x)dx $$
- **Variance:** the *second central moment* $$ V = \int^{\infty}_{-\infty} (x - \mu)^{2} h(x) dx $$
- **Standard deviation:** $$ \sigma = \sqrt{V} $$
- **Skewness:** related to the *third central moment* $$ \Sigma = \int^{\infty}_{-\infty} \left(\frac{x - \mu}{\sigma} \right)^{3} h(x) dx $$
- **Kurtosis:** related to the *fourth central moment* $$ K = \int^{\infty}_{-\infty} \left(\frac{x - \mu}{\sigma} \right)^{4} h(x) dx - 3$$
- **Absolute deviation about** $\mathbf{d}$**:** also known as the *mean deviation* when taken about the mean (ie., $d = \overline{x}$) $$ \delta = \int^{\infty}_{-\infty} |x - d| h(x) dx $$
- **Mode:** $$ \left(\frac{dh(x)}{dx} \right)_{x_m} = 0 $$
- **P% quantile:** $$ \frac{p}{100} = \int^{q_p}_{-\infty} h(x) dx $$
	- **Interquartile range:** the difference between the first and third quartiles, $q_{25}$ and $q_{75}$
##### ***Data-based Estimates of Descriptive Statistics***
- **Sample arithmetic mean:** $$ \overline{x} = \frac{1}{N} \sum^{N}_{i=1} x_i $$
- **Sample standard deviation:** $$ s = \sqrt{\frac{1}{N - 1} \sum^{N}_{i=1} (x_i - \overline{x})^{2}} $$
- **Estimators** $\overline{x}$ and $s$ can be judged by comparing their mean square errors (**MSEs**): $$ MSE = V + \text{bias}^{2} $$ where $V$ is the variance (the *measured error*) and the $bias$ is the expectation value of the difference of the estimator ($\overline{x}, s$) and its true population value ($\mu, \sigma$) (the *systematic error*)
	- **Consistent estimators:** estimators whose $V$ and $bias$ vanish as sample size goes to infinity
	- When $N$ is large and if $V$ of $h(x)$ is finite, then $\overline{x}$ and $s$ should be distributed around their sample values according the Gaussian distributions (**asymptotically normal**) with widths (**standard errors**) of: $$ \sigma_{\overline{x}} = \frac{s}{\sqrt{N}} \text{ and } \sigma_s = \frac{s}{\sqrt{2(N-1)}} = \frac{1}{\sqrt{2}} \sqrt{\frac{N}{N-1}} \sigma_{\overline{x}}$$
	- Estimators can be compared in terms of their **efficiency**, measuring how large of a sample is required to obtain a given accuracy
		- **Minimum variance bound (MVB):** smallest attainable variance for an unbiased estimator, a **minimum variance unbiased estimator (MVUE)**
	- Quantiles are more robust when determining location and scale parameters than mean and stdev.; the median and interquartile range are less affected by outliers; however it is easier to compute mean than median of a large sample
- **Gaussian width estimator:** determined from the interquartile range; an unbiased estimator of $\sigma$ for a "perfect" Gaussian distribution $$ \sigma_G = 0.7413 (q_{75} - q_{25}) $$
- **Standard error:** for an arbitrary quantile $q_p$ $$ \sigma_{qp} = \frac{1}{h_p} \sqrt{\frac{p(1-p)}{N}} $$ where $h_p$ is the value of the pdf at the $p$th percentile; depends on the underlying $h(x)$
	- For a Gaussian distribution, the standard error of the median is: $$ \sigma_{q50} = s \sqrt{\frac{\pi}{2N}} $$
- **Going forward:** common distributions used are called $p(x|I)$ to distinguish their mathematical definitions (equivalent to $h(x)$) from $f(x)$ estimated from data; parameters that describe the distribution $p(x)$ and "hidden" in $I$ ("information"); $p(x|I) dx$ is the probability of having a value of the random variable between $x$ and $x + dx$

### 3.3. Common Univariate Distribution Functions
##### ***The Uniform Distribution***
- Also the **top-hat** or **box** distribution; described by: $$ p(x|\mu,W) = \frac{1}{W} \text{ for } |x - \mu| \leq \frac{W}{2} $$ and 0 otherwise, where $W$ is the width of the "box"
- Descriptive Statistics:
	- Distribution is symmetric
	- Parameters: $\mu, W$
	- Mean: $\overline{x} = \mu$
	- Median: $q_{50} = \mu$
	- Mode: NA
	- Std: $\sigma = \frac{W}{\sqrt{12}}$
	- Gaussian width estimator: $\sigma_G = 0.371 W$
	- Skewness: $\Sigma = 0$
	- Kurtosis: $K = -1.2$
- Can arbitrarily vary location of distribution along the $z$-axis, and multiple $x$ by an arbitrary scale factor, without impacting distribution's shape
##### ***The Gaussian Distribution***
- Also the **normal** distribution, $\mathcal{N}(\mu,\sigma$); described by: $$ p(x|\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(\frac{-(x - \mu)^2}{2 \sigma^2} \right) $$
- Descriptive Statistics:
	- Distribution is symmetric
	- Parameters: $\mu, \sigma$
	- Mean: $\overline{x} = \mu$
	- Median: $q_{50} = \mu$
	- Mode: $x_m = \mu$
	- Std: $\sigma = \sigma$
	- Gaussian width estimator: $\sigma_G = \sigma$
	- Skewness: $\Sigma = 0$
	- Kurtosis: $K = 0$ (by definition)
	- Interquartile range: $q_{75} - q_{25} = \sigma 2 \sqrt{2} \text{erf}^{-1}(0.5) \approx 1.349 \sigma$
- The convolution of two Gaussian distributions is also Gaussian (it retains a Gaussian shape but its integral is not unity); the Fourier transform of a Gaussian is also Gaussian
- The sample mean and sample variance are independent
- The mean of samples drawn from an almost arbitrary distribution will follow a Gaussian
- The cdf of a Gaussian cannot be evaluated in closed form, and is usually expressed in terms of the **Gauss error function**, $\text{erf}(z)$
- The integral of $p(x|\mu, \sigma)$ between two arbitrary integration limits, $a$ and $b$, is also the difference between the two integrals $P(b|\mu, \sigma)$ and $P(a|\mu, \sigma)$
	- For $b = -a = \mu + M \sigma$, the integral equals $\text{erf}(M/\sqrt{2})$
	- For $M = 1,2,3$, integral equals $0.682, 0.954, 0.997$
- If $x$ follows $\mathcal{N}(\mu, \sigma)$, then $y = \exp (x)$ has a **log-normal** distribution
	- Mean: $\overline{x} = \exp(\mu + \sigma^2 / 2$
	- Median: $q_{50} = \exp(\mu)$
	- Mode: $x_m = \exp(\mu - \sigma^2)$
##### ***The Binomial Distribution***
- Describes the distribution of a variable that can take only two discrete values (0 or 1, success or failure, etc.)
- If $b$ is probability of success, then the distribution of a discrete variable $k$ that measures how many times success occurred in $N$ trials is given: $$ p(k|b,N) = \frac{N!}{k!(N-k)!} b^{k} (1 - b)^{N-k} $$
	- **Bernoulli distribution** when $N = 1$
- Descriptive Statistics:
	- Parameters: $b, N$
	- Mean: $\overline{k} = bN$
	- Std: $\sigma_{k} = [N b (1 - b)]^{1/2}$
- **Multinomial distribution:** a distribution of $M$ discrete variables $k_m$ that can have more than two discrete values; probability of each value is given by $b_m, b = 1, ..., M$
##### ***The Poisson Distribution***
- A special case of the binomial distribution; number of trials $N$ goes to infinity so that probability of success $p = k/N$ stays fixed, then number of successes $k$ is controlled by $\mu = pN$ and given by: $$ p(k|\mu) = \frac{\mu^{k} \exp(-\mu)}{k!} $$
- Descriptive Statistics:
	- Distribution is symmetric
	- Parameters: $\mu$
	- Mean: $\overline{x} = \mu $
	- Median: $q_{50} = \mu - 1/3$
	- Mode: $x_m = \mu - 1$
	- Std: $\sigma = \sqrt{\mu}$
	- Gaussian width estimator: NA
	- Skewness: $\Sigma = 1 / \sqrt{\mu}$
	- Kurtosis: $K = 1 / \mu$
- As $\mu$ increases, the Poisson distribution becomes more like a Gaussian $\mathcal{N}(\mu, \sqrt{\mu})$
	- Thus the Poisson distribution is sometimes called the **law of small numbers**, the **law of rare events**; where $p$ is small, not $\mu$
- Important to **Astronomy**: describes the distribution of number of photons counted in a given interval (Poisson noise, I think?)
##### ***The Cauchy (Lorentzian) Distribution***
- Distribution of a continuous variable; described by: $$ p(x|\mu, \gamma) = \frac{1}{\pi \gamma} \left(\frac{\gamma^2}{\gamma^2 + (x - \mu)^2} \right) $$
- Descriptive Statistics:
	- Distribution is symmetric
	- Parameters: $\mu \text{ (location parameter)}), \gamma \text{ (scale parameter})$
	- Median: $q_{50} = \mu$
	- Mode: $x_m = \mu$
	- Gaussian width estimator: $\sigma_G = 1.483\gamma$
	- Interquartile range: $q_{75} - q_{25} = 2\gamma$
	- $\overline{x}$, $V$, $\sigma$ and higher moments do not exist as tails decrease slowly $x^{-2}$ for large $|x|$
##### ***The Exponential (Laplace) Distribution***
- Described by: $$ p(x|\mu, \Delta) = \frac{1}{2 \Delta} \exp \left(\frac{- |x - \mu|}{\Delta} \right) $$
- Often only defined for $x > 0$ (**one-sided exponential distribution**), as opposed to the **double exponential** or **Laplace distribution**
- Simplest case of a one-sided exponential distribution is $p(x|\tau) = \tau^{-1} \exp(-x/\tau)$, where both mean and std $= \tau$; describes the time between two successive events which occur **continuously** and **independently** at a constant rate (eg. photons arriving at a detector)
	- Number of events during fixed time interval $T$ is given by the Poisson distribution with $\mu = T / \tau$
- Descriptive Statistics (for double exponential/Laplace):
	- Distribution is symmetric around $\mu$
	- Parameters: $\mu, \Delta$
	- Mean: $\overline{x} = \mu$
	- Median: $q_{50} = \mu$
	- Mode: $x_m = \mu$
	- Std: $\sigma = \sqrt{2} \Delta$
	- Gaussian width estimator: $\sigma_G = 1.028\Delta$
	- Skewness: $\Sigma = 0$
	- Kurtosis: $K = 3$
	- Interquartile range: $q_{75} - q_{25} = 2(\ln 2) \Delta$
- Differences from Gaussian distribution:
	- Std is larger that $\sigma_G$ ($\sigma \approx 1.38 \sigma_G$)
	- Tails distributions: $|x_i - \mu| > 5\sigma$ happen less than 1 per million cases for Gaussian; ~1 per thousand for exponential
	- Kurtosis in Gaussian is 0; is 3 in exponential
##### ***The*** $\mathbf{\chi^2}$ ***Distribution***
- If $\{x_i\}$ are drawn from a Gaussian and $z_i = (x_i - \mu) / \sigma$, then the sum of its squares, $Q = \Sigma^{N}_{i=1} z^{2}_{i}$, follows a $\chi^2$ distribution with $k=N$ degrees of freedom; described as: $$ p(Q|k) \equiv \chi^2 (Q|k) = \frac{1}{2^{k/2} \Gamma (k/2)} Q^{k/2 -1} \exp(-Q/2) \text{ for } Q > 0 $$ where $\Gamma$ is the gamma function, and for positive integers $k$, $\Gamma(k) \equiv (k - 1)!$
	- Distribution of $Q$ depends only on sample size $N$, not $\sigma$ or $\mu$
	- $Q$ is very sensitive to outliers
- Can also define distribution **per degrees of freedom** as: $$ \chi^{2}_{\text{dof}} (Q|k) \equiv \chi^2 (Q/k|k) $$
- Descriptive Statistics for $\chi^{2}_{\text{dof}}$:
	- Distribution is symmetric
	- Parameters: $k$
	- Mean: $\overline{x} = 1$
	- Median: $q_{50} = (1 - 2/9k)^3$
	- Mode: $x_m = \text{max } (0, 1 - 2/k)$
	- Std: $\sigma = \sqrt{2/k}$
	- Gaussian width estimator: NA
	- Skewness: $\Sigma = \sqrt{8/k}$
	- Kurtosis: $K = 12/k$
- As $k$ increases, $\chi^{2}_{\text{dof}}$ distribution tends to $\mathcal{N}(1, \sqrt{2/k})$
##### ***Student's $\mathbf{t}$ Distribution***
- Described by: $$ p(x|k) = \frac{\Gamma (\frac{k+1}{2})}{\sqrt{\pi k} \Gamma (\frac{k}{2})} \left(1 + \frac{x^2}{k} \right)^{- \frac{k+1}{2}} $$ where $k$ is the number of degrees of freedom; for $k=1$, this is a Cauchy distribution with $\mu = 0, \gamma = 1$
- Descriptive Statistics:
	- Distribution is symmetric and bell-shaped, but tails are heavier than a Gaussian
	- Parameters: $k$
	- Mean: $\overline{x} = 0$ for $k>1$; $\text{undefined}$ for $k=1$
	- Median: $q_{50} = 0$ for $k>1$; $\text{undefined}$ for $k=1$
	- Mode: $x_m = 0$ for $k>1$; $\text{undefined}$ for $k=1$
	- Std: $\sigma = \sqrt{k/(k - 2)}$ for $k>2$
	- Skewness: $\Sigma = 0$  for $k>3$
	- Kurtosis: $K = 6/(k - 4)$ for $k>4$
- For large $k$, distribution tends to $\mathcal{N} (0,1)$
- For a sample of $N$ measurements, $\{x_i\}$, drawn from a Gaussian $\mathcal{N}(\mu, \sigma)$, the variable $$ t = \frac{\overline{x} - \mu}{s / \sqrt{N}} $$ follows a Student's $t$ distribution with $k = N - 1$ degrees of freedom
	- Based on data-based estimates $\overline{x}$ and $s$, not true values
	- $t$ values of samples drawn from different Gaussian distributions will follow identical distributions as long as $N$ remains unchanged
##### ***Fisher's $\mathbf{F}$ Distribution***
- Described by: $$ p(x|d_1, d_2) = C \left(1 + \frac{d_1}{d_2} x \right)^{- \frac{d_1 + d_2}{2}} x^{\frac{d_1}{2} - 1} $$ for $x \geq 0$, $d_1 > 0$, $d_2 > 0$, and: $$ C = \frac{1}{B(d_1/2, d_2/2)} \left(\frac{d_1}{d_2} \right)^{d_1 / 2} $$ where $B$ is the beta function
- Descriptive Statistics (when both $d_1$ and $d_2$ are large):
	- Parameters: $d_1, d_2$
	- Mean: $\overline{x} = d_2 / (d_2 - 2) \approx 1$
	- Std: $\sigma = \sqrt{2 (d_1 + d_2) / (d_1 d_2)}$
- Describes the distribution of the ratio of two independent $\chi^{2}_{\text{dof}}$ variables with $d_1$ and $d_2$ degrees of freedom; useful in comparing std of two samples
##### ***The Beta Distribution***
- Described by: $$ p(x|\alpha, \beta) = \frac{\Gamma (\alpha + \beta)}{\Gamma (\alpha) \Gamma (\beta)} x^{\alpha - 1} (1-x)^{\beta - 1} $$ for $0 < x < 1$, and $\alpha > 0$ and $\beta > 0$
- Descriptive Statistics:
	- Parameters: $\alpha, \beta$
	- Mean: $\overline{x} = \alpha / (\alpha + \beta)$
- Used as a conjugate prior for the binomial distribution
##### ***The Gamma Distribution***
- Described by: $$ p(x|k, \theta) = \frac{1}{\theta^k} \frac{x^{k-1} e^{-x / \theta}}{\Gamma(k)} $$ for $0 < x < \infty$
- Used as a conjugate prior for several distributions including the exponential and Poisson distributions
##### ***The Weibull Distribution***
- Described by: $$ p(x|k, \lambda) \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k - 1} e^{-(x / \lambda)^k} $$ for $x \geq 0$
- Descriptive Statistics:
	- Parameters: $k, \lambda$
	- Mean: $\overline{x} = \lambda \Gamma (1 + 1/k)$
	- Median: $q_{50} = \lambda (\ln 2)^{1/k}$
- Shape parameter $k$ can interpolate between the exponential distribution ($k = 1$) and the **Rayleigh distribution** ($k = 2$); as $k$ tends to infinity, distribution becomes a **Dirac** $\mathbf{\delta}$ **function**
- Used to describe a random failure process with a variable rate, wind behaviour, distribution of extreme values, and size distribution of particles
- Cumulative distribution is described as: $$ H_W(x) = 1 - e^{-( / \lambda)^k} $$
	- Cdf based on data, $F(x)$, defines $z = \ln (-\ln (1 - F(x)))$
	- In the $z$ vs. $\ln x$ plane, distribution is a straight line with slope equal to $k$ and intercept equal to $-k \ln \lambda$
### 3.4. The Central Limit Theorem
- For an arbitrary distribution $h(x)$, characterised by $\mu$ and $\sigma$, the mean of $N$ values $\{x_i\}$ drawn from that distribution will approximate a Gaussian distribution $\mathcal{N}(\mu, \sigma / \sqrt{N})$, with the accuracy improving with $N$
- Provides the theoretical foundation for the practise of repeated measurements in order to improve the accuracy of the final results
	- We can average our measurements (computing their means) and expect the $1 / \sqrt{N}$ in accuracy regardless of details in our measuring apparatus
- **Caveat:** theorem makes strong assumptions about $h(x)$:
	- Must have a standard deviation
	- Must have a tail that falls off faster than $1 / x^2$ for large $x$
- Theorem cannot be applied if $h(x)$ has a **Cauchy distribution**, as they don't have a well-defined mean or std; Cauchy tails are extended and only decrease as $x^2$, not fast enough
- **Bernoulli's theorem:** also the **weak law of large numbers**; the sample mean converges to the distribution mean as sample size increases
	- Not applicable to distributions with ill-defined variance, ie. Cauchy again
- Application to the **uniform distribution**
	- Uni. has no tails but theorem is still applicable, however using the sample mean to estimate $\mu$ (with accuracy improving as $1 / \sqrt{N}$) is less efficient than another method
	- Best estimator for $\mu$ is the middle of the range of $x_i$ values: $$ \tilde{\mu} = \frac{\min(x_i) + \max(x_i)}{2} $$
		- $\tilde{\mu}$ is a more efficient estimator of location parameter $\mu$ than the mean value of $x_i$ for $N > 2$; accuracy improves as $1/N$
	- Best estimate for $W$ is: $$ \tilde{W} = [\max(x_i) - \min(x_i)] \frac{N}{N-2} $$
	- Width of the allowed range for $\mu$ is: $$ R = \frac{2W}{N} $$
	- Standard deviation of $\tilde{\mu}$ is: $$ \sigma_{\tilde{\mu}} = \frac{R}{\sqrt{12}} = \frac{2W}{\sqrt{12}N} $$

### 3.5. Bivariate and Multivariate Distribution Functions
##### ***Two-Dimensional (Bivariate) Distributions***
- Two values are measured in each instance: $N$ measured values of $x_i$ and $y_i$
- Example two-dimensional distribution $h(x,y)$ such that: $$ \int^{\infty}_{-\infty} \int^{\infty}_{-\infty} h(x,y) \text{ } dx \text{ } dy = 1 $$
- **Variances** of both values are: $$ V_x = \int^{\infty}_{-\infty} \int^{\infty}_{-\infty} (x - \mu_x)^2 h(x,y) dx dy $$
- **Means** of both values are: $$ \mu_x = \int^{\infty}_{-\infty} \int^{\infty}_{-\infty} x h(x,y) dx dy $$
- **Covariance** of $x$ and $y$, a measure of the dependence of the two variables on each other: $$ V_{xy} = \int^{\infty}_{-\infty} \int^{\infty}_{-\infty} (x - \mu_x) (y - \mu_y) h(x,y) dx dy $$
- **Standard deviations** are:
	- $\sigma_x = \sqrt{V_x}$
	- $\sigma_y = \sqrt{V_y}$
	- $\sigma_{xy} = V_{xy}$ (no square root)
- **Sum and variances:** for $z = x + y$, the variance of $z$, the sum, is: $$ V_z = V_x + V_y + 2 V_{xy} $$
- **Marginal distribution** of one variable, eg. for $x$: $$ m(x) = \int^{\infty}_{-\infty} h(x,y) dy $$
	- Different from the 2D distribution evaluated at $y = y_0$, $h(x, y_0)$
	- $m(x)$ is a normalised probability distribution; $h(x, y_0)$ is not
- **Uncorrelated variables:** if $\sigma_{xy} = V_{xy} = 0$
	- If they are also **independent**, then we can treat them as two 1D distributions, ie. $$ h(x,y) = h_x(x) h_y(y) $$
##### ***Bivariate Gaussian Distributions***
- Defining $I = (\mu_x, \mu_y, \sigma_x, \sigma_y, \sigma_{xy})$
- General case given by: $$ p(x,y| I) = \frac{1}{2\pi \sigma_x \sigma_y \sqrt{1 - \rho^2}} \exp \left( \frac{-z^2}{2(1 - \rho^2)} \right) $$ where $$ z^2 = \frac{(x - \mu_x)^2}{\sigma_x^2} + \frac{(y - \mu_y)^2}{\sigma_y^2} - 2\rho \frac{(x - \mu_x)(y - \mu_y)}{\sigma_x \sigma_y} $$ and correlation coefficient $$ \rho = \frac{\sigma_{xy}}{\sigma_x \sigma_y} $$
	- For perfectly correlated variables such that $y = ax + b$: $\rho = a/ |a| \equiv \text{sign}(a)$
	- For uncorrelated variables: $\rho = 0$
- The **population** correlation coefficient $\rho$ is directly related to Pearson's **sample** correlation coefficient $r$
- The **contours** in the ($x,y$) plane, defined by $P(x,y|I) = \text{constant}$ are ellipses centred on ($x = \mu_x, y = \mu_y$)
	- Angle $\alpha$ between the $x$-axis and the ellipses' major axis is: $$ \tan(2 \alpha) = 2\rho \frac{\sigma_x \sigma_y}{\sigma_x^2 - \sigma_y^2} = 2 \frac{\sigma_{xy}}{\sigma_x^2 - \sigma_y^2}$$
- **Principal axes** are the coordinate axes $P_1, P_2$: $$ P_1 = (x - \mu_x) \cos \alpha + (y - \mu_y) \sin \alpha $$ $$ P_2 = -(x - \mu_x) \sin \alpha + (y - \mu_y) \cos \alpha $$
	- The min and max **widths** obtainable for any rotation for the coordinate axes are: $$ \sigma_{1,2}^2 = \frac{\sigma_x^2 \sigma_y^2}{2} \pm \sqrt{\left( \frac{\sigma_x^2 - \sigma_y^2}{2} \right)^2 + \sigma_{xy}^2} $$
	- Can also define $\sigma_x, \sigma_y$ by $\sigma_{1,2}$ and $x,y$ by $P_{1,2}$ (eq. in text book, 3.85-3.88)
- Generally true that $p(x,y|I)$ evaluated for any fixed value $x$ will be proportional to a Gaussian width of $\sigma_{\ast}$, where $$ \sigma_{\ast} = \sigma_y \sqrt{1 - \rho^2} \leq \sigma_y $$
##### ***A Robust Estimate of Bivariate Gaussian Distribution from Data***
- Can find estimated sample values in similar ways to univariate distributions, ie. estimating ($\overline{x}, \overline{y}, s_x, s_y, s_{xy}$)
- Estimating principal axes with $\alpha$: $$ \tan(2\alpha) = 2 \frac{s_x s_y}{s_x^2 - s_y^2} r $$ where $\alpha$ is used for both population and sample values, and $r$ is the Pearson's sample correlation coefficient; estimates $s_x$ and $s_y$ are found from interquartile range, $r$ below
	- This estimation is greatly effected by outliers, better to use median over mean and interquartile range to estimate variance
- **Robustly estimating $r$**, use identity for correlation coefficient (here, the population one): $$ \rho = \frac{V_u - V_w}{V_u + V_w} $$ where $V$ is variance and the transformed coordinates are defined as: $$ u = \frac{\sqrt{2}}{2} \left(\frac{x}{\sigma_x} + \frac{y}{\sigma_y} \right) $$ and $$ w = \frac{\sqrt{2}}{2} \left(\frac{x}{\sigma_x} - \frac{y}{\sigma_y} \right) $$ with the covariance between them $V_{uw} = 0$
	- Replacing $V$ is robust estimator $\sigma_G^2$ gives estimate of sample $r$ instead of population $\rho$, and therefore estimate of axis angle $\alpha$
##### ***Multivariate Gaussian Distributions***
- Can continue adding dimensions by using the vector variable $\mathbf{x}$ (as opposed to scalar variable $x$ for univariate distributions)
	- Vector $\mathbf{x}$ has $M$ components (ie. dimensionality $M$)
	- In 1D: $x$ has $N$ values $x_i$
	- In MD: each $M$ component of $\mathbf{x}$ ($x^j, j=1, ..., M$) has $N$ values $x_i^j$
- Argument of exponential function of $p(x,y|I)$ from 3.5.2 is: $$ \arg = - \frac{1}{2} (\alpha x^2 + \beta y^2 + 2xy) $$
	- Allows the full probability to be written in **matrix notation:** $$ p(\mathbf{x}|I) = \frac{1}{(2\pi)^{M/2} \sqrt{det(\mathbf{C})}} \exp \left(- \frac{1}{2} \mathbf{x}^T \mathbf{Hx} \right) $$ where $\mathbf{x}$ is a column vector, $\mathbf{x}^T$ is its transposed row vector, $\mathbf{C}$ is the covariance matrix, and $\mathbf{H}$ is the inverse of $\mathbf{C}$ (also symmetric with positive eigenvalues)
	- Covariance matrix elements: $$ C_{kj} = \int^{\infty}_{-\infty} x^k x^j p(\mathbf{x}|I) \text{ } d^{M}x $$
	- Can also write argument of exponential function in component form: $$ \mathbf{x}^T \mathbf{Hx} = \sum^{M}_{k=1} \sum^{M}_{j=1} H_{kj} x^k x^j $$

### 3.6. Correlation Coefficients
##### ***Pearson's correlation coefficient** $\mathbf{r}$
- Sample coefficient: $$ r = \frac{\sum^{N}_{i=1} (x_i - \overline{x}) (y_i - \overline{y})}{\sqrt{\sum^{N}_{i=1} (x_i - \overline{x})^2} \sqrt{\sum^{N}_{i=1} (y_i - \overline{y})^2}} $$ for $-1 \leq r \leq 1$; $r=0$ for uncorrelated variables
- For pairs of ($x_i,y_i$) drawn from two uncorrelated univariate Gaussians, distribution of $r$ follows Student's $t$ distribution: $$ t = r\sqrt{\frac{N - 2}{1 - r^2}} $$
- A measured value of $r$ can be transformed into the **significance statements** that $\{x_i\}$ and $\{y_i\}$ are correlated, eg. "given $N$, the probability of a specific $r$ arises by chance is %"
- For bivariate Gaussian distributions with non-zero $\rho$, can use the Fisher transformation to estimate confidence interval for $\rho$ from measured $r$ value; distribution of $F$: $$ F(r) = \frac{1}{2} \ln \left( \frac{1+r}{1-r} \right) $$
	- Distribution follows a Gaussian with mean $\mu_F = F(\rho)$ and std $\sigma_F = (N-3)^{-1/2}$
- Main deficiencies:
	- The measurement errors for $\{x_i\}$ and $\{y_i\}$ are not used
	- **Sensitive to Gaussian outliers**; distribution doesn't follow Student's $t$ distribution if $\{x_i\}$ and $\{y_i\}$ aren't drawn from a bivariate Gaussian
##### ***Nonparameteric Correlation Tests***
- Two best known are **Spearman's correlation coefficient** $\mathbf{r_S}$ and **Kendall's correlation coefficient** $\tau$; both use concept of **ranks**, the index of $x_i$ in sorted data ($R_i^x$); advantage of ranks is that each value ($1, ..., N$) occurs only once
	- (Ignoring continuous variables that can be equal to each other)
- Spearman's $r_S$ defined analogously to Pearson's $r$ but with ranks instead of data values, alternatively given as: $$ r_S = 1 - \frac{6}{N(N^2 - 1)} \sum^N_{i=1} (R^x_i - R^y_i)^2$$
- Kendall's $\tau$ uses **concordant** and **discordant pairs** for comparison of ranks
	- For $\{x_i\}$ and $\{y_i\}$, comparing two pairs of values $j$ and $k$ will produce similar numbers of concordant pairs ($(x_j - x_k) (y_j - y_k) > 0$) and discordant pairs ($(x_j - x_k) (y_j - y_k) < 0$); for perfect correlation/anticorrelation, all $N(N-1)/2$ possible pairs will be concordant/discordant
	- Counting number of concordant ($N_c$) and discordant pairs ($N_d$) gives $\tau$: $$ \tau = 2 \frac{N_c - N_d}{N(N-1)} $$
	- $\tau$ is probability the two data sets are in the same order minus probability they aren't
- For **no correlation** and $N>10$: distribution of $\tau$ is approximately a Gaussian with $\mu=0$ and width: $$ \sigma_{\tau} = \left[\frac{2 (2N + 5)}{9N(N-1)} \right]^{1/2} $$
	- $\sigma_{\tau}$ gives the significance level of a given $\tau$; the probability that a large value would arise by chance with no correlation
- For **correlated** sets: for a bivariate Gaussian distribution of $x$ and $y$ with cc $\rho$, the expectation value of $\tau$ is: $$ \overline{\tau} = \frac{2}{\pi} \sin^{-1}(\rho) $$
	- $\tau$ is unbiased estimator of $\rho$, $r_S$ is not
	- Efficiency of $\tau$ relative to $r$ here is $> 90\%$, and can exceed it by a large factor for non-Gaussian distributions
- $\tau$ is a good general choice for measuring correlation of data sets; calculating $N_c$ and $N_d$ typically goes as $\mathcal{O} (N^2)$

### 3.7. Random Number Generation for Arbitrary Distributions
- **Monte Carlo simulations:** numerical simulations of the measurement process, uses to understand selection effects and resulting biases; involve drawing artificial or "mock" samples from specified distributions
	- Eg. drawing a random floating point number between 0 and 1 from a random distribution, called a **uniform deviate**
- Python and Numpy random number generators are based on the Mersenne twister algorithm
- For 1D cases:
	- **Transformation method:** for pdf $f(x)$ and cdf $F(x)$, use a uniform deviate generator to choose a value $0 \leq y \leq 1$, then chose $x$ such that $F(x) = y$
- For MD cases where distributions are inseparable:
	- For $h(x,y)$, first draw $x$ using marginal distribution equation; given $x$ (now $x_0$), the value of $y$ (called $y_0$) is generated by the properly normalised 1D cumulative conditional probability distribution in the $y$ direction: $$ H(y|x_0) = \frac{\int_{-\infty}^{y} h(x_0,y') dy'}{\int_{-\infty}^{\infty} h(x_0,y') dy'} $$
	- For multivariate Gaussians specifically, mock samples can be generated in the principal axes space, then values can be "rotated" to appropriate coord. system


Next chapter: [[Chapter 4]]
Extra terminology: [[NumPy Functions]], [[SciPy Functions]], [[AstroML Functions]]