Based on the assumptions we know that
$$
\begin{align*}
S' &= -\beta IS\\
I' &= \beta IS - \gamma I\\
R' &= \gamma I
\end{align*}
$$
If we add all three equations, we note that $S' + I' + R' = 0$, which is consistent with the assumption that $S + I + R = N$ (no births or deaths are happening in the population - no vital dynamics). As such, we only need to look at $S$ and $I$ to fully characterize the system,
$$
\begin{align*}
S' &= -\beta IS\\
I' &= \beta IS - \gamma I\\
\end{align*}
$$
Finding the equilibrium points, we note that there are infinitely many, but two we may want to look at more closely are $(S, I) = (N, 0)$ and $(S,I) = (\gamma/\beta, 0)$.

The first one, everyone is healthy (susceptible) not no disease. The second one, part of the pop remains susceptible and the rest have recovered.

* The ratio $\gamma/\beta$ is special. Why? Well, if we only have a small number of infected initially, then this  $I'$ decreases only when 
  $$
  I' < 0\\
  \Rightarrow \beta S < \gamma\\
  \Rightarrow S < \gamma/\beta
  $$
  Initially, this means the number of disease cases decreases if 
  $$
  \frac{\beta S}{\gamma} = \mathcal{R_0} < 1,
  $$
  where $\mathcal{R}_0$ is known as the *basic reproduction number*. In practice, the <a href="https://rt.live/"> effective reproductive number </a> may be more helpful (this takes into account the average number of new infections caused by single person in $S$)

Let's divide the differential equation for $I$ by the one for $S$. Then, we have
$$
\frac{I'}{S'}=\frac{dI}{dS} = \frac{\beta IS - \gamma I}{-\beta IS} = -\left(1 - \frac{\gamma}{\beta S}\right)
$$
Then, if $I \neq 0$, we have
$$
I = -S + \frac{\gamma}{\beta} \log S + c,
$$
with a constant of integration

Note that as $S \rightarrow 0$, we have $I \rightarrow -\infty$ , which is impossible in our case because the number of infected can be at minimum 0. This suggests that there is a point where $I = 0$ and we can solve for the right hand side