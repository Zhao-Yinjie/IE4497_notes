<h2> IE4497 formula</h2>


$$
\text{Gaussian Dist: } p(x|\mu, \sigma^2)= \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{1}{2\sigma^2}  (x - \mu)^2 \right)\\
\text{If } X \sim \mathcal{N}(\cdot|\mu,\sigma^2) \text{, then}\\
aX \sim \mathcal{N}(\cdot|a\mu,a^2\sigma^2); X+c \sim\mathcal{N}(\cdot|\mu+c,\sigma^2);
\text{let }Z \sim\mathcal{N}(\cdot|\mu,\sigma^2)\text{, then} X = \sigma Z+\mu \sim\mathcal{N}(\cdot|\mu,\sigma^2)\\
\text{if X and Y}\sim \mathcal{N}(\cdot|0,v^2) \text{are independent, then } X+Y \sim\mathcal{N}(\cdot|0,\sigma^2+v^2)\\
\text{If} X \sim \mathcal{N}(\cdot|\mu,\sigma^2)
Joint~pdf:P((X,Y)\in A)=\int_A p(x,y)dxdy,~~p(x,y)=p(x)p(y|x)~~(=p(x)p(y)~if~ independent)\\
p(x) = \int{p(x,y)dy} (marginal~distribution),~~p(x|y)=\frac{p(x,y)}{p(y)}=\frac{p(y|x)p(x)}{p(y)}(Bayes'~Theorem)\\
\frac{p(a|b,c)}{p(a|c)}=\frac{p(b|a,c)}{p(b|c)},
E[X]=\int xp(x)dx,\\
~E[g(X)]=\int g(x)p(x)dx,~E[X+Y]=E[X]+E[Y],~E[X|Y=y]=\int xp(x|y)dx\\
E[E[X|Y]]=E[X],~E[f(X)g(Y)|Y]=E[f(X)|Y]g(Y)\\
var(X) = E[(X-E[X])^2]=E[X^2]-E[X]^2,~cov(X,Y)=E[(X-E[X])(Y-E[Y])]\\
~var(X+Y)=var(X)+var(Y)+2cov(X,Y).~
\text{If } X = AZ+\mu: E[x]=\mu, cov(X) = AA^T\\
E[Ax]=AE[x],~cov(x)=\sum_{xx}=E[(x-E[x])(x-E[x])^T],~cov(Ax)=A~cov(x)A^T \\
cov(x,y)=\sum_{xy}=E[(x-E[x])(y-E[y])^T],
\frac{\delta(a^Tx)}{\delta x}=a,~\frac{\delta(x^TAx)}{\delta(x)}=(A+A^T)x,~\frac{\delta a^TXb}{\delta X}=ab^T\\
\frac{\delta{det(X)}}{\delta X}=det(X)(X^{-1})^T,~\frac{\delta a^TX^{-1}b}{\delta X}=-(X^{-1})^Tab^T(X^{-1})T\\
\underbrace{p(\theta |x)}_{posterior}=\frac{p(x,\theta)}{p(x)}=\frac{p(x|\theta)p(\theta)}{p(x)} \propto~\underbrace{p(x|\theta)}_{likelihood}.\underbrace{p(\theta)}_{prior}\\

\text{Residual sum of squares } RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2,

\text{Root mean squared error } RMSE = \sqrt{\frac{1}{n} RSS}.\\

\text{Coefficient of determination } R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{RSS}{\sum_{i=1}^n (y_i - \bar{y})^2},\\

\text{where } \bar{y} = \frac{1}{n} \sum_i y_i. \text{ TSS is the total sum of squares or the empirical variance of the data.}\\
\text{- Binomial distribution: } Bin (x | \theta, n) = \binom{n}{x} \theta^x (1 - \theta)^{n-x}, \binom{n}{x} = \frac{n!}{x!(n-x)!}\\
\text{- Beta distribution: } Beta (x | a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1} (1 - x)^{b-1} \text{ (*Beta dist is a conjugate prior to Bin dist)}\\
\text{- Categorical distribution: } Cat (x | \theta_1, \ldots, \theta_K) = \theta_x\\
\text{- Multinomial distribution: } Mult (x | \theta, n) = \binom{n}{x_1, \ldots, x_K} \prod_{k=1}^K \theta_k^{x_k},\binom{n}{x_1, \ldots, x_K} = \frac{n!}{x_1! \cdots x_K!}. \\
\text{- Dirichlet distribution: } Dir (x | \alpha_1, \ldots, \alpha_K) = \frac{\Gamma(\sum_{k=1}^K \alpha_k)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K x_k^{\alpha_k-1}, 
\text{ (*Dirich dist is a conjugate prior to Multi Nor dist)}\\
\text{- Joint Gaussian distribution: } N (x | \mu, \Sigma) = \frac{1}{(2\pi)^{K/2}(\det \Sigma)^{1/2}} exp\left(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu)\right)\\
\text{- Uniform distribution: } Unif (x | a, b) =\begin{cases}
\frac{1}{b-a} &\text{if } a \leq x \leq b,\\
0 &\text{otherwise}.
\end{cases},
\text{- Exponential distribution: } Exp (x | \lambda) =\begin{cases}
\lambda e^{-\lambda x} &\text{if } x \geq 0,\\\
0 &\text{otherwise}.
\end{cases}\\
\text{Bernoulli distribution:}  p(D|\theta) = \theta^{N_1}(1-\theta)^{N_0}, 
\text{N}_k = \sum_{i=1}^n 1\{x_i = k\}.

\text{Exponential distribution:}  p(D|\lambda) = \lambda^n \exp(-\lambda \sum_{i=1}^n x_i) \\

\text{Normal distribution:}  p(D|\theta) = N(\mu,\sigma^2)^n = \frac{1}{(2\pi)^{n/2}\sigma^n} \exp \left( -\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2 \right) \\

\text{Linear model:}  y = w^\top x + \epsilon,

\text{Basis function expansion:}  y = w^\top \phi(x) + \epsilon,

\text{MLE of linear model:}  w_{ML} = (\Phi^\top \Phi)^{-1}\Phi^\top y \\

\text{Kullback-Leibler divergence:}  D(p_{\theta_0}\|p_\theta) = E_{\theta_0}\left[\log \frac{p(x|\theta_0)}{p(x|\theta)}\right],

\text{MAP estimate:}  \theta_{MAP} = \max_\theta p(\theta|D) \\

\text{MMSE estimate:} \hat{\theta}(D) = \min_{a} E[(\theta - a)^2 | D] = E[\theta | D]\\

\text{Mixture model:} \quad p(x | \theta) = \sum_{k=1}^K \pi[k]p_k(x | \theta),

\text{GMM:} \quad p(x | \theta) = \sum_{k=1}^K \pi[k]\mathcal{N}(x | \mu_k, \Sigma_k), \quad \theta = (\pi[k], \mu_k, \Sigma_k)_{k=1}^K \\

\text{E step:} \quad Q(\theta | \theta^{(m)}) = E_{y\sim p(\cdot|x,\theta^{(m)})}\left[\log p(y | \theta) | x, \theta^{(m)}\right],

\text{M step:} \quad \theta^{(m+1)} = \arg\max_{\theta\in\Theta} Q(\theta | \theta^{(m)}) \\

\text{GMM E step:} \quad r^{(m)}_{ik} = p(z_i = k | x_i, \theta^{(m)}) = \frac{\pi^{(m)}[k]\mathcal{N}(x_i | \mu^{(m)}_k, \Sigma^{(m)}_k)}{\sum_{k'}\pi^{(m)}[k']\mathcal{N}(x_i | \mu^{(m)}_{k'}, \Sigma^{(m)}_{k'})}, 

n^{(m)}_k = \sum_{i=1}^n r^{(m)}_{ik}, \\

Q(\theta | \theta^{(m)}) = -\frac{1}{2\sigma^2}\sum_{i=1}^n \|x_i - \mu^{(m)}_{k_i}\|^2 + const. \\

\text{GMM M step:} \quad
\pi^{(m+1)}[k] = n^{(m)}_k / n,

\mu^{(m+1)}_k = (1/n^{(m)}_k)\sum_{i=1}^n r^{(m)}_{ik} x_i;~~

\Sigma^{(m+1)}_k = (1/n^{(m)}_k)\sum_{i=1}^n r^{(m)}_{ik}(x_i - \mu^{(m+1)}_k)(x_i - \mu^{(m+1)}_k)^{\top}. \\

\text{MAP estimate:} \quad
\theta^{\text{MAP}} = \arg\max_{\theta\in\Theta} (\log p(x | \theta)+ log p(\theta)). ~~

\text{EM for MAP:}
E step: Q(\theta | \theta^{(m)}) = E_{y\sim p(\cdot|x,\theta^{(m)})}\left[\log p(y | \theta) | x, \theta^{(m)}\right] \\
M step: 	\theta^{(m+1)} = arg max
θ∈θ
(Q(\theta | θ
(m)
)+ log p(\theta)) \\

\text{Markov property: } p(x_t | x_1, \ldots, x_{t-1}) = p(x_t | x_{t-1}),
\text{Transition probability: } T(i,j) = p_{x_t| x_{t-1}}(j|i)\\
\text{Unigram model: } p(x_t = x),
\text{Bigram model: } p(x_t | x_{t-1}),
\text{n-gram model: } p(x_t | x_{t-1}, x_{t-2}, \ldots, x_{t-n+1})\\
\text{PageRank score: } \pi_i = \sum_j T(j,i) \pi_j\\
\text{MLE for Markov model: } 
\log p(D | \pi, T) = \sum_{i=1}^n \log \pi(x_i,0) + \sum_{i=1}^n \sum_{t=1}^{t_i} \log T(x_{i,t-1}, x_{i,t}) \\
= \sum_{x=1}^M N_x \log \pi(x) + \sum_{x=1}^M \sum_{y=1}^M N_{xy} \log T(x, y)\\
N_x = \sum_{i=1}^n \mathbb{I}\{x_{i,0} = x\},
N_{xy} = \sum_{i=1}^n \sum_{t=1}^{t_i} \mathbb{I}\{x_{i,t-1} = x, x_{i,t} = y\}.
\hat{\pi}(x) = \frac{N_x}{n}, \hat{T}(x,y) = \frac{N_{xy}}{\sum_z N_{xz}},\\
\text{HMM: } p(x_0, \ldots, x_T, z_0, \ldots, z_T | \theta) = \pi(z_0) p(x_0 | z_0) \prod_{t=1}^T T(z_{t-1}, z_t) p(x_t | z_t)\\
\text{Baum-Welch algorithm: MLE - } \log p(D | \theta) = \sum_{i=1}^n \log \pi(z_{i,0}) + \sum_{i=1}^n \sum_{t=1}^{t_i} \log T(z_{i,t-1}, z_{i,t}) + \sum_{i=1}^n \sum_{t=0}^{t_i} \log p(x_{i,t} | \phi_{z_{i,t}})\\
\text{Steps for BW Algo: }
(1) \text{Initialize } \theta^{(0)}.
(2) \text{E step: At iteration } m, \text{ use Forward-Backward Algorithm to compute}\\

\gamma_{i,t}(z) = p(z_{i,t} = z | x_{i,\cdot}, \theta^{(m)}) \propto \alpha_j(z)\beta_j(z),~~

\xi_{i,t}(z, z') = p(z_{i,t-1} = z, z_{i,t} = z' | x_{i,\cdot}, \theta^{(m)})\\
\propto \alpha_{t-1}(z)p(x_{i,t} | z_{i,t} = z')\beta_t(z')p(z_{i,t} = z' | z_{i,t-1} = z).~~
(3) \text{M step: Find } (m + 1).\\

\pi^*(z) = \frac{\sum_{i=1}^n \gamma_{i,0}(z)}{n},~~

T^*(z, z') = \frac{\sum_{i=1}^n \sum_{t=1}^{t_i} \xi_{i,t}(z, z')}{\sum_u \sum_{i=1}^n \sum_{t=1}^{t_i} \xi_{i,t}(z, u)},

\hat{\phi}_z = \text{emission prob. model parameters.}\\


\text{Viterbi algorithm: } z_0^*, \ldots, z_T^* = \arg\max_{z_0,\ldots,z_T} p(z_0,\ldots,z_T | x_0,\ldots,x_T)\\

\text{Cumulative distribution function:}

P(X \leq x) = F(x),

X = F^{-1}(U), \text{where } U \sim \text{Unif}([0, 1]), u \leq F(x) \Leftrightarrow F^{-1}(u)\leq x \\


\text{Transformation method:} 

p_Y(y) = \sum_{k=1}^K \frac{p_X(x_k)}{|f'(x_k)|}, \text{where } x_1, x_2, \ldots, x_K \text{ are solutions to } f(x) = y \\


\text{Rejection sampling:} 

p(z) = \frac{1}{M} \tilde{p}(z), \text{where } M \text{ is unknown},

kq(z) \geq \tilde{p}(z) \text{ for all } z \\

\text{Accept } z \sim q(z) \text{ if } u \sim \text{Unif}([0, kq(z)]) \leq \tilde{p}(z) \\

\text{Acceptance probability: }

P(z \text{ accepted}) = \int P(z \text{ accepted} | z)q(z) dz

= \int \frac{\tilde{p}(z)}{kq(z)} q(z) dz

= \frac{M}{k}\\

\text{Rejection sampling for Bayesian Inference: }

\tilde{p}(\theta) = p(D | \theta)p(\theta) \text{ and } q(\theta) = p(\theta):

k = \max_{\theta} \frac{\tilde{p}(\theta)}{q(\theta)} = \max_{\theta} p(D | \theta)\\

\text{Importance sampling:} 
\text{ sample } z \text{ where } |f(z)|p(z) \text{ is large for better efficiency rather than from } p(z) \text{ directly.}\\

E_p[f(z)] = E_q\left[\frac{p(z)}{q(z)} f(z)\right],~~ 

\tilde{w} = \frac{p(z)}{q(z)}, w(z) = \frac{\tilde{w}(z)}{\sum_{i=1}^n \tilde{w}(z_i)},~~

E_p[f(z)] \approx \sum_{i=1}^n w(z_i) f(z_i) \\

\text{Tail sampling: } 

P(X > a) \approx \sum_{i=1}^n w(z_i),

\text{where } z_1, z_2, \ldots \text{ are sampled from } q(z) \text{ with support } (a, \infty)~~

\text{and } w(z_i) = \frac{p(z_i)}{q(z_i)} \\

\text{Sampling importance resampling (SIR):} ~~

1. \text{Sample } z_1, \ldots, z_n \text{ from } q(z). \\
2. \text{Compute weights } w(z_1), \ldots, w(z_n) \text{ where } w(z_i) = \frac{\tilde{w}(z_i)}{\sum_{j=1}^n \tilde{w}(z_j)}. \\
3. \text{Resample with replacement from } \{z_1, \ldots, z_n\} \text{ according to weights } (w(z_1), \ldots, w(z_n)).\\


\text{SIR for Bayesian inference: }

w(z_i) = \frac{p(D | z_i)}{\sum_{j=1}^n p(D | z_j)};~~

\text{Sampling for EM: }

Q(\theta | \theta^{(m)}) \approx \frac{1}{n} \sum_{i=1}^n \log p(x, z_i | \theta)\\

\text{Stationary distribution:}

\sum_x \pi(x)T(x,y) = \pi(y),~~

\text{Reversible MC: } 

\pi(x)T(x,y) = \pi(y)T(y,x) \\

\text{Metropolis-Hastings algorithm:}~~

(1) \text{Initialize } x = Z_0. ~
(2) \text{For each } m = 1, 2, \ldots ~
(3) \text{Sample } y \sim q(x,y). \\
(4)  \text{Compute acceptance probability } A(x,y) = \min \left( 1, \frac{\tilde{\pi}(y)q(y,x)}{\tilde{\pi}(x)q(x,y)} \right). \\
\text{if } \pi(x) \propto \psi(x)h(x) \text{ and } q(x,y) = h(y) \text{, then }
A(x,y) = \min \left( 1, \frac{\psi(y)}{\psi(x)} \right).\\
(5)  \text{With probability } A(x,y), \text{ set } Z_m = y; \text{ otherwise set } Z_m = x ~
(6)  \text{Update } x = Z_m. \\

 \text{ - random walk MH: }q(x, y) = q(y - x).~~

y - x \sim \mathcal{N}(\cdot | 0, \Sigma) \text{ - Gaussian centered at } x.\\

y - x \sim \text{Unif}[-\delta, \delta]^d \text{ - Uniform distribution centered at } x.\\

\text{Gibbs sampling:} ~~

p(z_i | z_{-i}) = \frac{p(z_1, \ldots, z_d)}{p(z_{-i})} .~~


\text{For each } i, \text{ let } z_{-i} = \{z_1, \ldots, z_{i-1}, z_{i+1}, \ldots, z_d\}, \text{ i.e., } z_i \text{ removed.}\\

(1) \text{Initialize } (z_1^{(0)}, \ldots, z_d^{(0)}).~~
(2) \text{For each } k:~~

(3) \text{sample } z_1^{(k)} \text{ from } p(\cdot | z_2^{(k-1)}, \ldots, z_d^{(k-1)})\\
(4) \text{sample } z_2^{(k)} \text{ from } p(\cdot | z_1^{(k)}, z_3^{(k-1)}, \ldots, z_d^{(k-1)})~~
\ldots~~
(5) \text{sample } z_j^{(k)} \text{ from } p(\cdot | z_1^{(k)}, \ldots, z_{j-1}^{(k)}, z_{j+1}^{(k-1)}, \ldots, z_d^{(k-1)})
 ~~\ldots\\
(6) \text{sample } z_d^{(k)} \text{ from } p(\cdot | z_1^{(k)}, \ldots, z_{d-1}^{(k)})\\
\text{Ising model: } 
\text{Given a noisy image y, want to recover z.}
\text{compute the posterior p(z | y)}.\\
Likelihood: p(y \mid z) = \prod_{j} p(y_j \mid z_j) = \prod_{j} \mathcal{N}(y_j \mid z_j, \sigma^2)\\

p(z_j \mid z_{-j}) \propto \prod_{s \in N_j} \psi(z_s, z_j),\text{where }N_j \text{is the neighorhood of pixel }z_j\text{, and}\\
\psi(u, v) = \exp(Juv) \text{with J > 0 as the"coupling strength"}.\\
$$


