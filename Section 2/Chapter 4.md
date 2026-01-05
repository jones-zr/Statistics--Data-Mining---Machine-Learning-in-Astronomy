*Classical Statistical Inference*
### 4.1. Classical vs Bayesian Statistical Inference
- **Three types of inference:**
	- **Point estimation:** What is the best estimate for a model parameter $\theta$ based on available data
	- **Confidence estimation:** How confident should we be in our point estimate
	- **Hypothesis testing:** Are the data consistent with a given hypothesis/model
- **Classical** or **frequentist** tenants:
	- Probabilities are **relative frequencies of events**; are objective properties of the real world
	- Parameters are **fixed, unknown constants**; they don't fluctuate, probability statements about them are meaningless
	- Statistical procedures should have **well-defined, long-run** frequency properties
- **Bayesian** tenants:
	- Probability describes the **degree of subjective belief**; probability statements can be made about data, model parameters and models etc.
	- Probability distributions quantify the uncertainty of our knowledge about parameters
- Both are concerned with uncertainties about estimates; main difference is which parts of the situation have a certain probability value
- Classical and Bayesian results are often the same; Bayesian approach has more computational difficulties

### 4.2. Maximum Likelihood Estimation (MLE)
##### ***The Likelihood Function***
- A quantitative description of the measuring process; given a known or assumed behaviour of the measuring apparatus (or population distribution in statistics), the likelihood of observing any given value can be calculated
- The likelihood of a data set $\{x_i\}$ drawn from an $\mathcal{N}(\mu, \sigma)$ parent distribution is $L$, the product of likelihoods/probabilities of each value: $$ L \equiv p(\{x_i\}| M(\theta)) = \prod_{i=1}^{n} p(x_i|M(\theta)) $$where $M$ is the model, the understanding/assumptions of the measurement process with $k$ parameters $\theta_p, p = 1, ..., k$
- The likelihood of each individual $x_i$ is a true pdf, but the product of them all is **no longer normalised** to 1
- $L$ can be considered a function of both the data $x_i$ and a function of the model $M$
##### ***The Maximum Likelihood Approach***
1. Formulate the data likelihood for some model $M$, $p(D|M)$; amounts to an assumption about how data is generated; models are described with set of parameters $\theta$
2. Search for best model parameters which maximise $p(D|M)$; yields the MLE **point estimates** $\theta^0$
3. Determine the confidence region for model parameters using mathematical or numerical techniques such as bootstraping, jack-knifing or cross-validation
4. Perform **hypothesis tests** to draw conclusions about models and point estimates
##### ***The MLE Applied to a Homoscedastic Gaussian Likelihood***
- **Log-likelihood function:** $$ \ln L \equiv \ln [L(\theta)] $$
	- Maximum of likelihood function occurs at same place as for log-likelihood function; allows for ignoring constants in likelihood function
- For this example where $D = \{x_i\}$ are drawn from a Gaussian with mean $\mu$ and "measurement error" $\sigma$, the is only one model parameter, $\mu$ ($k=1$ and $\theta_1 = \mu$)
- The value of $\mu$ that maximises $\ln L$ (the MLE $\mu^0$) is determined with the condition $$ \left. \frac{d \ln L(\mu)}{d \mu} \right|_{\mu^0} \equiv 0 $$
- Log-likelihood of this Gaussian is derived as: $$ \ln L(\mu) = \text{constant} - \sum_{i=1}^{N} \frac{(x_i - \mu)^2}{2 \sigma^2} $$therefore MLE of $\mu$ is: $$ \mu^0 = \frac{1}{N} \sum_{i=1}^{N} x_i $$
	- In this example, $\mu^0$ simplifies to the arithmetic mean of $\{x_i\}$
##### ***Properties of Maximum Likelihood Estimators***
- A critical assumption: the data truly comes from the specified model class/distribution
- MLEs are **consistent** estimators; the converge to the true parameter value as $N$ increases
- MLEs are **asymptotically normal** estimators; estimator distribution approaches a normal distribution centred at the MLE
- MLEs asymptotically achieve the theoretical minimum possible variance (the Cramér-Rao lower bound); they achieve the **best possible error** given the provided data
##### ***The MLE Confidence Intervals***
- Using the asymptotic normality of MLE, the error matrix on the MLE is computed from the covariance matrix as: $$ \sigma_{jk} = ([F^{-1}]_{jk})^{1/2} $$where $$ F_{jk} = - \left. \frac{d^2 \ln L}{d\theta_j d\theta_k} \right|_{\theta = \theta_0} $$
- Related to the **expected Fisher information**, the expectation value of the second derivative of $-\ln L$ wrt. $\theta$; inverse of the Fisher information gives a lower bound on variance (the Cramér-Rao lower bound)
- When error matrix is evaluated at $\theta_0$, gives **observed Fisher information**
- Diagonal elements, $\sigma_{ii}$, correspond to marginal error bars for parameters $\theta_i$
	- If $\sigma_{jk} = 0$ for $j \neq k$: inferred values for parameters $\theta_j$ and $\theta_k$ are uncorrelated; error in one parameters has no effect on other parameter
	- If $\sigma_{jk} \neq 0$ for $j \neq k$: errors for parameters $\theta_j$ and $\theta_k$ are correlated
- For a Gaussian, the errors of the MLE (ie. the uncertainty of the mean) are: $$ \sigma_{\mu} = \left( \left. - \frac{d^2 \ln L(\mu)}{d \mu^2} \right|_{\mu^0} \right)^{-1/2} = \frac{\sigma}{\sqrt{N}} $$as expected
##### ***The MLE Applied to a Heteroscedastic Gaussian Likelihood***
- Analogously to homoscedastic Gaussian, log-likelihood of a heteroscedastic Gaussian is derived as: $$ \ln L(\mu) = \text{constant} - \sum_{i=1}^{N} \frac{(x_i - \mu)^2}{2 \sigma^2} $$ and MLE of $\mu$ is: $$ \mu^0 = \frac{\sum_{i}^{N} w_i x_i}{\sum_{i}^{N} w_i} $$with weights $w_i = \sigma_i^{-2}$; $\mu_0$ is a **weighted** arithmetic mean of all measurements
- The uncertainty of $\mu^0$ is: $$ \sigma_{\mu} = \left( \sum_{i=1}^{N} \frac{1}{\sigma_i^2} \right)^{-1/2} = \left( \sum_{i=1}^{N} w_i \right)^{-1/2} $$
	- When all $\sigma_i$ are equal, $\mu^0$ and $\sigma_{\mu}$ reduce to those of a homoscedastic Gaussian as above
##### ***The MLE in the Case of Truncated and Censored Data***
- **Truncated data**: data for which the **selection probability** or **selection function** $S(x)$ is $0$; we know nothing about this data, including whether they exist or not
	- eg. $S(x) = 0$ for $x > x_{max}$ and $x < x_{min}$; data is truncated outside of this range
	- The data likelihood of a single datum must be a properly normalised pdf; truncation effects enter analysis through a **renormalisation constant**
- **Censored data**: measuring data for an existing source was attempted but the value is outside of some known interval, eg. upper and lower limits of measurements
- For an example Gaussian with truncated data such that $S(x) = 1$ for $x_{min} \leq x \leq x_{max}$ and $S(x) = 0$ otherwise:
	- Likelihood for a single data point is: $$ p(x_i | \mu, \sigma, x_{min}, x_{max}) = C(\mu, \sigma, x_{min}, x_{max}) \frac{1}{\sqrt{2\pi} \sigma} \exp \left(\frac{-(x_i - \mu)^2}{2 \sigma^2} \right) $$where the renormalisation constant is: $$ C(\mu, \sigma, x_{min}, x_{max}) = (P(x_{max}| \mu, \sigma) - P(x_{min} | \mu, \sigma))^{-1} $$ where $P$ is the cdf for the Gaussian
	- The log-likelihood is therefore: $$ \ln L(\mu) = \text{constant} - \sum_{i=1}^{N} \frac{(x_i - \mu)^2}{2 \sigma^2} + N \ln [C(\mu, \sigma, x_{min}, x_{max})] $$
		- Identical to normal Gaussian with new term to account for truncation that does not depend on data as $x_{min}$ and $x_{max}$ are the same for all points
	- There is no simple closed-form expression for $\mu^0$ in this case
##### ***Beyond the Likelihood: Other Cost Functions and Robustness***
- Cost functions quantify some "cost" associated with parameter estimation; expectation value of the cost function is called **risk**
- One risk, the **mean integrated square error** (MISE) is defined as: $$ \text{MISE} = \int_{-\infty}^{\infty} [f(x) - h(x)]^2 dx $$
	- Shows how close the empirical estimate $f(x)$ is to the true pdf $h(x)$
	- Based on the cost function given by the mean square error ($L_2$ norm); a cost function that minimises absolute deviation is called the $L_1$ norm
		- Eg. a MLE applied to a Gaussian likelihood gives a $L_2$ cost function; applied to a Laplace distribution gives a $L_1$ cost function
- Cost functions important in cases where formalising likelihood function is difficult; optimal solution can still be found by minimising corresponding risk

### 4.3. The Goodness of Fit and Model Selection
##### ***The Goodness of Fit for a Model***
- Can find the maximum value of the likelihood of a data set, $L^0$ or $\ln L^0$, and try to determine how likely it is that this value would have arisen by chance
- Gaussian likelihood can be rewritten as: $$ \ln L = \text{constant} - \frac{1}{2} \sum_{i=1}^{N} z_i^2 = \text{constant} - \frac{1}{2} \chi^2 $$where $z_i = (x_i - \mu)/\sigma$; therefore the distribution of $\ln L$ can be determined from the $\chi^2$ distribution with $k=1, k_1 = \mu$
	- $\chi^2$ distribution doesn't depend on actual values of $\mu$ and $\sigma$; expectation value is $N - k$, std is $\sqrt{2(N-k)}$
- For a **good fit**, expect that: $$ \chi_{\text{dof}}^2 = \frac{1}{N - k} \sum_{i=1}^{N} z_i^2 \approx 1 $$
	- If $(\chi_{\text{dof}}^2 - 1)$ is many times larger than the std, it is **unlikely** that the data were generated by the assumed model; outliers may significanly increase $\chi_{\text{dof}}^2$
	- If model and measurement errors are consistent, $\chi_{\text{dof}}^2$ will be close to $1$
	- if measurement errors are **overestimated**, $\chi_{\text{dof}}^2$ will be improbably **low**
	- If measurement errors are **underestimated**, $\chi_{\text{dof}}^2$ will be improbably **high**; high $\chi_{\text{dof}}^2$ can also indicate that the model is insufficient to fit the data
##### ***Model Comparison***
- Assessing the probability of $L^0$ with a $\chi^2$ distribution is only possible when the likelihood is Gaussian; else different models can be ranked by $L^0$ values with the largest value being the best, however this alone does not tell how well a model fits the data
	- Scoring system also needs to account for model complexity, degrees of freedom, and penalise models for additional parameters not supported by data
- **Akaike information criterion** (AIC): a popular general classical method for model comparison
	- Based on asymptotic approximation, effective for simple models
	- AIC is computed as: $$ \text{AIC} \equiv -2 \ln (L^0 (M)) + 2k + \frac{2k(k + 1)}{N - k - 1} $$where $k$ is is number of model parameters and $N$ is number of data points
	- Model with smallest AIC is the best model to select; for equally successful models, model with fewer free parameters is best

### 4.4. ML Applied to Gaussian Mixtures: The Expectation Maximisation Algorithm
##### ***Gaussian Mixture Model***
- here : D
##### ***Class Labels and Hidden Variables***
##### ***The Basics of the Expectation Maximisation Algorithm***

### 4.5. Confidence Estimates: The Bootstrap and the Jackknife

### 4.6. Hypothesis Testing
##### ***Simple Classification and Completeness vs. Contamination Trade-Off***

### 4.7. Comparison of Distribution
##### ***Regression toward the Mean***
##### ***Nonparametric Methods for Comparing Distributions***
##### ***Comparison of Two-Dimensional Distributions***
##### ***Is My Distribution Really Gaussian?***
##### ***Is My Distribution Bimodal***
##### ***Parametric Methods for Comparing Distributions***

### 4.8. Nonparametric Modelling and Histograms
##### ***Histograms***
##### ***How Do We Determine the Histogram Errors?***

### 4.9. Selection Effects and Luminosity Function Estimation
##### ***Lynden-Bell's $\mathbf{C^{-}}$ Method***
##### ***A Flexible Method of Estimating Luminosity Functions***

### 4.10. Summary


Next chapter: [[Chapter 5]]
New terminology: