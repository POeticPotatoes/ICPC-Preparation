## Table of Contents
#### Math
Polynomials
- Newton's Method
- Horner's Method
- Appendix


## Math
### Polynomials
<table style="width:100%;height:3000;">
<tr><td style="width:50%">

#### Newton's Method (Root Finding)
Starting with a value of x **close to the root**,  
Using:

$x_{n+1} = x_n-\frac{f(x_n)}{f'(x_n)}$

Repeat until $x(n)$ converges to find a root
</td><td>

#### Horner's Method (Evaluation)
Express $\sum^{n}_{i=0}{a_ix^n}$ as

$a_0 + x(a_1 + x(a_2 + x(a_3 +...))))$

<br>

Evaluated in $n$ operations (O(n))

</td></tr>
<tr><td style="width:50%">

#### Rational Root Theorem (Root Finding)
If the roots for a polynomial $\sum^{n}_{i=0}{a_ix^n}$ are **rational**, then they must be some combination of: 

$\pm\frac{f(a_0)}{f(a_n)})$, where $f(x)$ represents all integer factors of a number $x$

Therefore trying all combinations of these numbers will produce all rational roots for a polynomial.
</td>
<td>

#### Newton-Horner Method (Root Finding)
Newton's Method can be used *in tandem* with Horner's method with the following steps:  
1.  Compute $p'(x_n)$ using the Power Rule or Ruffini's Rule
2. Evaluate $p(x_n)$ and $p'(x_n)$ using Horner's Method
3. Repeat until convergence
4. Once one root is found, use Ruffini's Rule to "deflate" the polynomial
5. Repeat until all roots are found
</td>
</tr>
<tr><td colspan=2>

#### Ruffini's Rule (Polynomial Division)
Any polynomial $p(x)$ can be expressed in terms of $(x-a)$ in the form 

$p(x) = q(x)(x-a) + p(a)$

where $q(a)$ is the quotient and can be found with synthetic division (essentially brute force) of $p(x)$ against $(x-a)$

**Example**

$\begin{aligned}
p(x)&=x^5-8x^4-72x^3+382x^2+727x-2310\\
    &=q(2)(x-2) + p(2)\\
    &=[(0+1)x^4+(1*2-8)x^3+(-6*2-72)x^2+(-84*2+382)x\\
        &\ \ + (214*2+727)](x-2) + 0\\
    &=(x^4-6x^3-84x^2+214x+1155)(x-2) + 0
\end{aligned}$

This can be used to "deflate" a polynomial (eliminate a root)

Furthermore, since:

$\begin{aligned}
p'(x)&=\frac{d}{dx}(q(x)(x-a) + p(a))\\
     &=q(x)(x-a)' + q'(x)(x-a)\\
     &=q(x) + q'(x)(x-a)
\end{aligned}$

We have:

$\begin{aligned}
p'(a)&=q(a) + q'(a)(a-a)\\
     &=q(a)
\end{aligned}$

In other words, this can also be used to evaluate the derivative of a polynomial *without the power rule* and can be used together with the Newton-Horner method.
> This only works when q(x) is derived from the *same* value of a

</td></tr>
</table>

### Appendix
**Polynomial Expansion (Brute Force)**
```c++
/*
 / deg represents the degree of the polynomial
 / Assume s[] is 1-based and stores all (sorted) roots
*/
int deg, p;
const int N = 110;
ll s[N], dp[2][N];

ll expand_polynomial() {
    memset(dp[0], 0, sizeof(dp[p]));
    dp[0][1] = 1; p =1;
    for (int i=1;i<=deg;i++, p^=1) {
        memset(dp[p], 0, sizeof(dp[p]));
        for (int j=i;j>=0;j--)
            dp[p][j+1] += dp[p^1][j+1]*s[i] + dp[p^1][j];
    }
    return p^1;
}
```
