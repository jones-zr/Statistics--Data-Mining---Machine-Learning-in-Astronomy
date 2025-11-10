*Bayesian Statistical Inference*
### 5.1. Introduction to the Bayesian Method
##### ***The Essence of the Bayesian Idea***
- Thesis: probability statements are not limited to data; can be made for model parameters and models themselves; inferences made by producing pdfs; model parameters are treated as random variables
- Both classical and Bayesian stats use data likelihood functions; Bayesian extends concept with extra information (a **prior**) and by assigning pdfs to all model parameters and models themselves
- Bayesian model motivated by its ability to provide a full probabilistic framework for data analysis; it incorporates unknown or uninteresting model parameters (**nuisance parameters**) in data analysis
- **Bayes' theorem**, obtained by applying Bayes' rule to a likelihood function $p(D|M)$, is: $$ p(M|D) = \frac{p(D|M) p(M)}{p(D)} $$
	- Essentially, combining an initial belief with new data to arrive at an improved belief; "improved belief" ($p(M|D$) is proportional to the product of "initial belief" ($p(M)$) and the probability that that belief generated the observed data ($p(D|M)$)
	- More explicitly, with the inclusion of prior information $I$ and model parameters $\boldsymbol{\theta}$: $$ p(M, \boldsymbol{\theta} | D, I) = \frac{p(D|M, \boldsymbol{\theta}, I) p(M, \boldsymbol{\theta} | I)}{p(D|I)} $$ where
		- $p(M, \boldsymbol{\theta} | D, I)$ is the **posterior** pdf for model $M$ and parameters $\boldsymbol{\theta}$, given data $D$, and prior information $I$
		- $p(D|M, \boldsymbol{\theta}, I)$ is the **likelihood** of the data given a model, fixed-value parameters describing it, and prior information
		- $p(M, \boldsymbol{\theta} | I)$ is the priori joint probability for the model and its parameters in the absence of data to compute a likelihood; the **prior**; can be expanded as: $$ p(M, \boldsymbol{\theta} | I) = p(\boldsymbol{\theta} | M, I) p(M|I) $$ and the integral of the prior $p(\boldsymbol{\theta} | M, I)$ over all parameters should be unity
		- $p(D|I)$ is the **probability of data**, or the prior predictive probability for $D$; provides the proper normalisation for the posterior pdf
- Controversial point of Bayesian method: the posterior, $p(M, \boldsymbol{\theta} | D, I)$, is not a probability in the same sense as the likelihood $p(D|M, \boldsymbol{\theta}, I)$ is
	- The posterior corresponds to the state of our knowledge/belief about a model and its parameters given data and prior information; is a posterior pdf for models and model parameters
	- When more data are taken, the posterior based on the first data set can be used as the prior for the second analysis; updating belief with new data
- Statistical 95% confidence regions = Bayesian **credible regions**
##### ***Steps of a Bayesian Statistical Inference***
1. Formulate the data likelihood $p(D|M, I)$
2. Chose the prior $p(\boldsymbol{\theta} | M, I)$, which incorporates all other knowledge that might exist but is not used when computing the likelihood
3. Determine the posterior pdf $p(M | D, I)$; often is properly normalised, so removes need to define $p(D|I)$
4. Search for the best model parameters $M$ which maximise $p(M|D,I)$, yielding the **maximum a posteriori** (MAP) estimate (an analogue of the **point estimate** from classical statistics)
	- Another natural Bayesian estimator, the **posterior mean**: $$ \overline{\theta} = \int \theta p(\theta|D) d\theta $$
5. Quantification of uncertainty in parameter estimates using **credible regions**
6. **Hypothesis testing** that incorporates the prior

### 5.2. Bayesian Priors
- The terms prior and posterior do not have absolute meanings
##### ***Priors Assigned by Formal Rules***
- **Informative priors:** priors that incorporate information from other measurements
- **Uninformative priors:** priors with no other information other than the data to be analysed; incorporate weak but still objective information; even the most uninformative priors still affect the estimates
- **Flat priors:** priors such that $p(\theta|I) \propto C$, where $C > 0$ is a constant
	- Sometimes considered to be ill defined as a flat prior on a parameter doesn't imply a flat prior on a transformed version of the parameters
- **Improper priors:** priors where the integral of their likelihoods does not equal unity; eg. $\int p(\theta|I) d\theta = \infty$ is not a pdf
	- Generally not a problem as long as the resulting posterior is a well-defined pdf
- **Principle of indifference:** a set of basic mutually exclusive possibilities need to be assigned equal probabilities, eg. each side of a d6 has a prior probability of 1/6
- **Principle of consistency:** the prior for a location parameter should not change with translations of the coordinate system, and yields a flat prior; also should not depend on choice of measurement units; these are **scale-invariant priors**
##### ***The Principle of Maximum Entropy***
- **Entropy:** measures the information content of a pdf; symbol $S$
	- Resembles thermodynamical entropy on purpose; units are $nat$ for natural unit
- Given a pdf defined by $N$ discrete values $p_i$ with $\sum_{i=1}^{N} p_i = 1$, its entropy is: $$ S = -\sum_{i=1}^{N} p_i \ln(p_i) $$
	- Continuous case given by: $$ S = -\int_{-\infty}^{\infty} p(x) \ln \left( \frac{p(x)}{m(x)} \right) dx $$ where the measure $m(x)$ are the values that would be assigned to $p_i$ in the case when no addition information is known; ensures entropy is invariant under a change of variables
- The **Principle**: when assigning uninformative priors, maximising entropy over a suitable set of pdfs will give the distribution that is the least informative
- In cases where only the mean and variance of the prior are known, and the distribution defined over the whole real line, the maximum entropy solution is a Gaussian with that mean and variance
- The **Kullback-Leibler (KL) divergence** from $p(x)$ to $m(x$) is: $$ \text{KL} = \sum_{i} p_i \ln \left( \frac{p_i}{m_i} \right) $$
	- Used the measure the information gain when moving from a prior distribution to a posterior distribution
	- Sometimes called the KL distance between two pdfs
##### ***Conjugate Priors***
- The special name for a prior that has the same functional form as the posterior probability
- Eg. when the likelihood function is a Gaussian $\mathcal{N}(\overline{x},s)$, the conjugate prior is also a Gaussian $\mathcal{N}(\mu_p, \sigma_p)$, and therefore the posterior function is also also a Gaussian $\mathcal{N}(\mu^0, \sigma^0)$ with: $$ \mu^0 = \frac{\mu_p / \sigma_p^2 + \overline{x} / s^2}{1 / \sigma_p^2 + 1 / s^2} \text{ and } \sigma^0 = (1 / \sigma_p^2 + 1 / s^2)^{-1/2} $$
- Most frequently encountered conjugate pairs are the beta distribution for binomial likelihood, and the gamma distribution for Poissonian likelihood
##### ***Empirical and Hierarchical Bayes Methods
- Normal Bayes approach: parameters of priors are chosen before any data are observed
- **Empirical Bayes** approach: parameters of priors (**hyperparameters**) are estimated from the data; set to their most likely values instead of integrated out; also known as the **maximum marginal likelihood**
- **Hierarchical Bayes** or multilevel approach: prior distributions (**hyperpriors**) depend on unknown variables/hyperparameters that describe the population level probabilistic model

### 5.3. Bayesian Parameter Uncertainty Quantification
##### ***Posterior Intervals***
- To obtain a Bayesian **credible region**, must find $a$ and $b$ such that: $$ \int_{-\infty}^{a} f(\theta) d\theta = \int_{b}^{\infty} f(\theta) d\theta = \alpha/2 $$
- Probability that true value of $\theta$ is between $a$ and $b$ is $1 - \alpha$ (analogous to classical confidence intervals)
- Interval ($a,b$) is the $\mathbf{1 - }\boldsymbol{\alpha}$ **posterior interval**
- Often the posterior pdf $p(\theta)$ is not an analytical function; can only be evaluated numerically; can approximate the $1-\alpha$ posterior interval through the $\alpha /2$ and ($1 - \alpha / 2$) sample quantiles of simulations/sampling
##### ***Marginalisation of Parameters***
- **Marginalisation:** integrating the posterior pdf over only uninteresting (**nuisance**) parameters; resulting pdf is the **marginal posterior pdf**
- If the value of the nuisance parameter is unknown, integrate over all plausible values to obtain the marginalised posterior pdf for the interesting parameter $x$
- The marginalised pdfs span a wider range of $x$ than posterior pdfs for $x$ where the value of the nuisance parameter is known

### 5.4. Bayesian Model Selection
- **Odds ratio:** comparing the posterior probabilities of two models $M_1$ and $M_2$ to find out which is better supported by data; favours $M_2$ over $M_1$ with: $$ O_{21} \equiv \frac{p(M_2 | D,I)}{p(M_1 | D,I)} $$
	- These two posteriors $M_{1,2}$ can be be obtained from the posterior pdf $p(M, \boldsymbol{\theta} | D, I)$ using **marginalisation** over the model parameter space $\boldsymbol{\theta}$
	- Posterior probability the model $M$ given data $D$ is: $$ p(M | D,I) = \frac{p(D | M,I) p(M|I)}{p(D|I)} $$ where $$ E(M) \equiv p(D | M,I) = \int p(D | M, \boldsymbol{\theta}, I) p(\boldsymbol{\theta} | M,I) d\boldsymbol{\theta} $$ is the **marginal likelihood** or **evidence** for model $M$
	- $E(M)$ is also called **global likelihood** for model $M$; is a **weighted average** of the likelihood function
	- To compute $p(D|I)$: $$ O_{21} = \frac{E(M_2)}{E(M_1)} \frac{p(M_2 | I)}{p(M_1 | I)} = B_{21} \frac{p(M_2 | I)}{p(M_1 | I)} $$ where $B_{21}$ (the **Bayes factor**) is the ratio of global likelihoods and is: $$ B_{21} = \frac{\int p(D | M_2, \boldsymbol{\theta}_2, I) p(\boldsymbol{\theta}_2 | M_2, I) d\boldsymbol{\theta}_2}{\int p(D | M_1, \boldsymbol{\theta}_1, I) p(\boldsymbol{\theta}_1 | M_1, I) d\boldsymbol{\theta}_1} $$
##### ***Bayesian Hypothesis Testing***
- Special case of model comparison where $M_2 = \overline{M_1}$, the complementary hypothesis to $M_1$
- $M_1$ is taken to be the null hypothesis; if data supports $M_2$ over $M_1$, then the null hypothesis is rejected
- With equal priors $p(M_1|I) = p(M_2|I)$, the odds ratio is: $$ O_{21} = B_{21} = \frac{p(D|M_2)}{p(D|M_1)} $$
	- It's not possible to compute $p(D|M_2)$ given $M_2$ is just the complementary hypothesis to $M_1$
- Bayesian hypothesis testing is unable to reject a hypothesis if there is no alternative explanation for observed data, as it's based on the posterior rather than the data likelihood
	- Classical version is based on data likelihood, and the null hypothesis can be rejected if it doesn't provide a good description of the data
##### ***Occam's Razor***
- The principle of selecting the simplest model that is in fair agreement with the data
- Odds ration has the ability to penalise complex models with many free parameters; Occam's Razor is naturally included in Bayesian model comparison
##### ***Information Criteria***
- **Bayesian information criterion (BIC)** is closely related to the odds ratio and similar to the AIC
- The BIC for model $M$ is: $$ BIC \equiv -2 \ln [L^0 (M)] + k \ln N $$ where $k$ is the number of model parameters, $N$ is the amount of data points, and $L^0 (M)$ is the maximum value of the data likelihood
	- The BIC of a model corresponds to $-2 \ln [E(M)]$
- When two models are compared via their BIC, the model with the smallest value wins; if they are equal, the model with the fewer free parameters wins
- The BIC penalises additional model parameters more harshly than the AIC; in general it is better to compute the odds ratio when computationally feasible

### 5.5. Nonuniform Priors: Eddington, Malmquist, and Lutz-Kelker Biases
- **Selection bias** or **Malmquist bias:** the difference between true distribution $h(x)$ and estimate $f(x)$ caused by sample truncation
- Similarly named **Eddington-Malmquist bias:** bias in brightness/magnitude measurements due to combined effect of measurement errors and nonuniform $h(x)$
	- Mathematically identical to **Lutz-Kelker bias**
- Main difference: Eddington-Malmquist bias and Lutz-Kelker bias disappear when measurement error for $x$ vanishes, **Malmquist bias** does not
- True and observed distributions are related via convolution: $$ f(x) = h(x) \star e(x) = \int_{-\infty}^{\infty} h(x') e(x - x') dx' $$ where $e(x)$ is a known error distribution
	- If $e(x) = \delta (x)$ then $f(x) = h(x)$; if $h(x) = \text{constant}$ then $f(x) = \text{constant}$
- For homoscedastic and Gaussian errors, $\Delta x = x_{obs} - x_{true}$ (broadly); can be expressed in terms of measured $f(x)$: $$ \Delta x = -\sigma^2 \frac{1}{f(x)} \frac{df(x)}{dx} $$ when evaluated at $x = x_{obs}$
	- $\Delta x$ vanishes for $\sigma = 0$; Eddington-Malmquist and Lutz-Kelker biases become negligible for vanishing errors

### 5.6. Simple Examples of Bayesian Analysis: Parameter Estimation
##### ***Parameter Estimation for a Gaussian Distribution***
- Estimating $\mu$:
	- For a set of $N$ measurements $\{x_i\}$ with heteroscedastic errors $\sigma_i$, the posterior pdf for $\mu$ is Gaussian in cases **when $\sigma_i$ are known**, regardless of data set size $N$; this is not true when $\sigma$ is unknown and determined from data
	- **When $\sigma$ is not known a priori**, the assumption of the Gaussian uncertainty of $\mu$ is valid only in the **large N limit**; when $N$ is not large, the posterior pdf for $\mu$ follows Student's $t$ distribution
	- Irrespective of the size of a data set, only $N$, $\overline{x}$, and $V$ are needed to fully capture the entire information content of the **$\mu$ posterior pdf**: $$ L_p \equiv \ln [p(\mu | \{x_i\}, I)] = \text{constant} - (N + 1) \ln \sigma - \frac{N}{2 \sigma^2} ((\overline{x} - \mu)^2 + V) $$where $V = (N - 1) s^2 / N$; $s$ is the sample standard deviation
- Estimating $\sigma$:
	- Analogous to determining $p(\mu | \{x_i\}, I)$, the posterior pdf of $\sigma$ is derived using marginalisation as: $$ p(\sigma | \{x_i\}, I) \propto \frac{1}{\sigma^N} \exp \left( \frac{-NV}{2\sigma^2} \right) $$
- For a Gaussian distribution with Gaussian errors:
	- If errors are homoscedastic, the resulting distribution of measurements is Gaussian
	- If errors are heteroscedastic, "" is not itself a Gaussian
##### ***Parameter Estimation for the Binomial Distribution***
- Estimating $b$:
	- **When $N$ is large**,  and it's (assumed Gaussian) uncertainty are determined as in [[Chapter 3#3.3. Common Univariate Distribution Functions#***The Binomial Distribution***|Binomial Distribution]]
	- **For small $N$**, the posterior pdf for $b$ is: $$ p(b | \{x_i\}) = C b^k (1 - b)^{N - k} $$ where $k$ is the actual observed number of successes in data set of $N$ values; $C$ is a normalisation constant
		- Maximum posterior occurs at $b_0 = k/N$
		- Standard error for $b_0$ is: $$ \sigma_b = \left[ \frac{b_0 (1 - b_0)}{N} \right]^{1/2} $$
##### ***Parameter Estimation from the Cauchy (Lorentzian) Distribution***
- The mean value for many independent samples will themselves follow the same Cauchy distribution, and will not benefit from the central limit theorem
- Estimating $\mu$ and $\gamma$:
	- Logarithm of the posterior pdf is: $$ L_p \equiv \ln [p(\mu, \gamma | \{x_i\}, I)] = \text{constant} + (N + 1) \ln \gamma - \sum_{i=1}^{N} \ln [\gamma^2 + (x_i - \mu)^2] $$
	- As sample size $N$ increases, both posterior marginal distributions become **asymptotically Gaussian**
	- The median and $\sigma_G$ can be used as a good shortcut to determine the best-fit parameters
##### ***Beating $\sqrt{N}$ for Uniform Distribution***
- With no tails, extreme values of $x_i$ are a better way to estimate location parameter than the mean value; errors improve with sample size $1/N$ rather than $1/\sqrt{N}$
- Estimating $\mu$ and $W$:
- Within $[W_{min}, W_{max}]$, the posterior pdf  $p(\mu, W | {x_i}, I)$ is proportional to $1/W^{N+1}$ without a dependance on $\mu$, and is 0 otherwise
##### ***Parameter Estimation for a Gaussian and a Uniform Background***
- For a mixed Gaussian and uniform distribution in some interval $W$, for known location parameter $\mu$: $$ L_p \equiv \ln[p(A, \sigma | {x_i}, \mu, W)] = \sum_{i=1}^{n} \ln \left[ A \frac{\exp \left( \frac{-(x_i - \mu)^2}{2\sigma^2} \right)}{\sqrt{2 \pi} \sigma} + \frac{1-A}{W} \right] $$
- Model example: a spectral line of known central wavelength $\mu$ but unknown width $\sigma$ and strength/amplitude $A$
	- Expect a covariance between $\sigma$ and $A$; an error for $\sigma$ is compensated for by a proportional error in $A$; if additional information were available about either parameter, the other would be better constrained
##### ***Regression Preview: Gaussian vs. Poissonian Likelihood***
- For data that are pairs of random variables, eg. $(x_1, y_1), ..., (x_M, y_M)$, we want to estimate $a$ and $b$ for a model $y=ax+b$ (**regression**)
- Data set ${x_i}$ drawn from $p(x) = ax + b$; since $p(x)$ must be normalised, $a$ and $b$ are related via: $$ b = \frac{1}{W} = ax_{1/2} $$ therefore for $a>0$: $$ L_p (a | {x_i}, x_{min}, x_{max}) = \sum_{i=1}^{N} \ln \left[ a (x_i - x_{1/2}) + \frac{1}{W} \right] $$
- For $y_i$, the expectation value and standard deviation are the same for a Poisson and Gaussian distribution; only the distribution shape changes
##### ***A Mixture Model: How to Throw Out Bad Data Points***
- Standard methods for estimating parameters are based on the assumption of Gaussianity
	- When the underlying distribution is known, maximise the posterior pdf
	- When model is not known a priori, use Bayesian framework to construct a model in terms of unknown nuisance model parameters, then marginalise over them to estimate quantities

### 5.7. Simple Examples of Bayesian Analysis: Model Selection
##### ***Gaussian or Lorentzian Likelihood?***
- Obtain model evidence by integrating the product of the data likelihood and the prior pdf for the model parameters, eg. $$ E(M = \text{Cauchy}) = \int p(\{x_i\} | \mu, \gamma, I) p(\mu, \gamma | I) d\mu d\gamma $$ and $$ E(M = \text{Gaussian}) = \int p(\{x_i\} | \mu, \sigma, I) p(\mu, \sigma | I) d\mu d\sigma $$
- If no other information is available, can assume the ratio of model priors $p(M_C |I) / p(M_G |I) = 1$, and thus the odds ratio is the same as the Bayes factor, $$ O_{CG} = \frac{E(M = \text{Cauchy})}{E(M = \text{Gaussian})} $$
- If the odds ratio is very close to 1, the comparison is inconclusive; the ability to discriminate between models in creases with number of data values
- Additionally, the presence of even a single outlier can have a large effect on the computed likelihood and conclusions
##### ***Understanding Knuth's Histograms***
- The best piecewise constant model has the number of bins $M$ which maximises: $$ F(M | \{x_i\}, I) = N \log M + \log \left[ \Gamma \left(\frac{M}{2}\right) \right] - M \log \left[ \Gamma \left(\frac{1}{2}\right) \right] - \log \left[ \Gamma \left(N + \frac{M}{2}\right) \right] + \sum_{k=1}^{M} \log \left[ \Gamma \left(n_k + \frac{1}{2}\right) \right] $$ where $n_k$ is the number of measurements $x_i$ which are found in bin $k$
- By assumption, bin width is constant and number of bins is the result of model selection
- The expectation value of the posterior pdf of $h_k$ is: $$ h_k = \frac{n_k + \frac{1}{2}}{N + \frac{M}{2}} $$ as the assumed prior distribution effectively places one half of a datum in each bin
- Optimal bin width varies depending on underlying distribution; for a Gaussian ($N$ up to $10^6$) shows: $$ \Delta_b = \frac{2.7 \sigma_G}{N^{1/4}} $$
	- With $\sigma_G$, this is applicable to non-Gaussian distributions if they don't have complex structure (ie. multiple nodes, extended tails)
- For non-Gaussian distributions, Scott's rule greatly underestimates the optimal number of histogram bins
- **Bayesian blocks**: the data are segmented into blocks, with the borders between two blocks being set by **changepoints**; the **log-likeliness fitness function** can be defined for each block: $$ F(N_i, T_i) = N_i (\log N_i - \log T_i) $$ where $N_i$ is the number of points in block $i$, and $T_i$ is the width of block $i$
	- These adaptive bin widths lead to a better representation of the underlying data
	- Statistical significance can be attached to the bin configuration
##### ***One Gaussian or Two Gaussians?***
- ...

### 5.8. Numerical Methods for Complex Problems (MCMC)
- Generic Monte Carlo methods are very inefficient, especially with high-dimensional integrals
- **Markov Chain Monte Carlo** methods return a sample of points, or chain, from the $k$-dimensional parameter space with a distribution that is asymptotically proportional to $p(\theta)$
	- With a Markov chain, quantitative description of the posterior pdf becomes and density estimation problem
##### ***Markov Chain Monte Carlo***
- **Markov chain**: a sequence of random variables where a given value nontrivially depends only on its preceding value; "memoryless" chain
- **Markov process**: process of generating a Markov chain as: $$ p(\theta_{i+1} | \{\theta_i\}) = p(\theta_{i+1} | \theta_i) $$
- To reach an equilibrium or stationary distribution of positions, it is sufficient that the transition probability is **symmetric**: $$ p(\theta_{i+1} | \theta_i) = p(\theta_i | \theta_{i+1}) $$
	- The **detail balance** or **reversibility condition**; the probability of a jump between two points does not depend on the direction of the jump
##### ***MCMC Algorithms***
- 
##### ***PyMC3: MCMC in Python***
##### ***Example: Model Selection with MCMC***
##### ***Example: Gaussian Distribution with Unknown Gaussian Errors***
##### ***Example: Unknown Signal with an Unknown Background***

### 5.9. Hierarchical Bayesian Modelling

### 5.10. Approximate Bayesian Computation

### 5.11. Summary of Pros and Cons for Classical and Bayesian Methods


Next chapter: [[Chapter 6]]
New terminology: [[Astropy Functions]]