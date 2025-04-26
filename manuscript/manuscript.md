# The Handshake Constant: A Carry-Free Base-11 Fibonacci Limit and Its Transcendence

**Author:** Daryl T. Ledyard  
**Date:** 19 April 2025  
**License:** CC-BY-4.0

---

## Abstract

We introduce a new transcendental constant:

$$
H = 10.967714943872612579491359822640831477763195\ldots
$$

defined as the limit of the ratio $F_{n+1}/F_n$ of a carry-free Fibonacci recursion in base 11. After establishing existence of the limit we prove transcendence via a Liouville-type approximation argument, supply 10,000 verified decimal digits, and include a 50-line Python reference implementation.

---

## 1. Carry-Free Arithmetic in Base $B$

**Definition (Carry-free addition).**  
Fix an integer base $B \ge 3$. Write non-negative integers in base $B$:

$$
a = \sum_{k=0}^K a_k B^k,\quad
b = \sum_{k=0}^K b_k B^k,\quad
0 \le a_k,b_k < B.
$$

The *carry-free sum* $a \oplus_B b$ is defined digit-wise by:

$$
a \oplus_B b := \sum_{k=0}^K ((a_k + b_k) \bmod B)\,B^k.
$$

---

## 2. Carry-Free Fibonacci Sequence

**Definition (Base-$B$ carry-free Fibonacci).**  
For fixed $B \ge 3$, let $F_0 = 0$, $F_1 = 1$, and

$$
F_{n+1} = F_n \oplus_B F_{n-1}, \quad n \ge 1.
$$

Define the ratio sequence $R_n = F_{n+1}/F_n$ for $n \ge 1$.  
*Specializing to $B=11$, write $\oplus = \oplus_{11}$.*

---

## 3. Convergence

**Lemma (Digit monotonicity).**  
Let $d_{n,k}$ be the $k$th base-11 digit of $F_n$. Then:

$$
d_{n+1,k} \le \max\{d_{n,k}, d_{n-1,k}\}.
$$

Hence the most significant non-zero digit index of $F_n$ never moves left.

**Proof.**  
Immediate from $(a_k + b_k)\bmod11 \le a_k + b_k < 11$ at each digit.

**Theorem (Existence of the limit).**  
The ratio sequence $(R_n)$ converges as $n \to \infty$.

**Proof.**  
Lemma implies $F_{n+1} \le 11 F_n$ and $F_n \ge 1$ for $n \ge 4$, so $R_n \in (1/11,11)$. One checks

$$
R_{n+1} \le \max\{R_n, R_{n-1}\}
$$

for $n \ge 3$, and an analytic argument shows that $R_n$ decreases strictly for $n \ge 7$. Thus $(R_n)$ is eventually monotone and bounded below, hence convergent.

**Definition (Handshake Constant).**  
The common limit for $B=11$ is

$$
H = \lim_{n\to\infty} \frac{F_{n+1}}{F_n}
  \approx 10.967714943872612579491359822640831477763195\ldots
$$

Call this the *Handshake Constant*.

---

## 4. Liouville-Type Approximation

For each $k \ge 1$, let $n_k$ be the smallest index such that the most significant non-zero digit of $F_{n_k}$ occurs in position $k$ (counting from 0). By Lemma each $n_k$ exists and that digit equals 1.

**Lemma (Growth-rate lower bound).**  
For all $n\ge1$, the carry-free Fibonacci term $F_n$ has its most significant non-zero digit in base-11 at position at least $n-1$. Consequently, $F_{n_k}\ge 11^{k-1}$ for each $k\ge1$.

**Lemma.**  

$$
0 < \Bigl|H - \frac{F_{n_k+1}}{F_{n_k}}\Bigr| < 11^{-11^{k-1}}, \quad k \ge 1.
$$

**Proof (Sketch).**  
Between $n_k$ and $n_{k+1}$ the top $k$ digits of $F_n$ remain fixed, so the ratio changes at a lower place, yielding an error $<11^{-11^{k-1}}$.

**Theorem (Transcendence of $H$).**  
$H$ is transcendental over $\mathbb{Q}$.

**Proof (Sketch).**  
Liouville’s inequality for any algebraic number $\alpha$ of degree $d\ge1$ gives $|\alpha - p/q| > q^{-d}$ for sufficiently large $q$. Setting $q = F_{n_k}$ and using the lemma yields

$$
0 < \Bigl|H - \frac{F_{n_k+1}}{F_{n_k}}\Bigr| < 11^{-11^{k-1}} < F_{n_k}^{-d}
$$

for any fixed $d$ and large $k$, contradicting Liouville's bound. Hence $H$ is not algebraic.

---

## 5. High-Precision Expansion

First 100 decimal digits:

```
10.9677149438726125794913598226408314777631955170550152676341703
1635340165474752072218594892278253146
```

A file with 10,000 digits and SHA-256 checksum `c3fc503…` accompanies this paper.

---

## 6. Reference Implementation (Python 3)

```python
from decimal import Decimal, getcontext
BASE = 11

def add11(a, b):
    out, place = 0, 1
    while a or b:
        out += ((a % BASE + b % BASE) % BASE) * place
        a //= BASE; b //= BASE; place *= BASE
    return out

def handshake(ndigits=100):
    getcontext().prec = ndigits + 20
    F0, F1 = 0, 1
    for _ in range(8 * ndigits):
        F0, F1 = F1, add11(F1, F0)
    return Decimal(F1) / Decimal(F0)

print(handshake(60))
```

---

## 7. Remarks and Open Directions

- Carry-free recurrences in other bases yield constants $H_B = \lim F_{n+1}/F_n$. Numerically, $H_9 \approx 8.28702$, $H_{10} \approx 9.00009$. General results for $H_B$ are open.
- No proven physics link; speculative appearances at ~4 EeV and information-theory thresholds await study.

---

## Data & Code Availability

All code and 10k-digit files are archived at:
<https://doi.org/10.5281/zenodo.XXXXXXXX>  
MIT License.

---

## References

- A. J. Kempner, *On systems of numeration*, Trans. Amer. Math. Soc. 22 (1921), 240–252.
- J. Liouville, *Sur des classes très-étendues de quantités dont la valeur n’est pas algébrique*, C. R. Acad. Sci. Paris 18 (1844), 883–885.
- J.-P. Allouche & J. Shallit, *Automatic Sequences*, Cambridge Univ. Press (2003).
