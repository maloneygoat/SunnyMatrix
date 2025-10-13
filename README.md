Within `Su(n)ny.jl` and Spin Wave Theory (SWT) there is the ability to calculate matrix elements and dynamical spin strucutre factor (DSSF). This is done within 'Su(n)ny.jl'
using different tricks and processes to speed up the calculations. I am interested in looking at Linear and Non-linear magnetic responses, which requires some adjustment to
the code. This repository is to store the attempts at making it work and different examples.

Linear response, or the linear magnetic suceptibility, is the simplest. It has the following formula:

$$\chi^{(1)}_{\alpha\beta}(t_1) = \frac{i}{2N} \left\langle \left[ M^\alpha(t_1), M^\beta(0) \right] \right\rangle$$

where $\alpha$, $\beta$ are the components of the magentisation. $M^\alpha = \sum_iS_i^\alpha(t)$, where $S_i^\alpha$ is spin operator component $\alpha$ on site
$i$ at time $t$. $N$ is the number of sites for renormalisation. 
In the special case for uniform fluctuations, the lattice wavevector $\vec{q} = 0$, and the above linear suceptibility is just the DSSF

$$S^{\alpha\alpha}(\mathbf{q}, \omega) = \sum_n \left| \langle n | S^\alpha_{-\mathbf{q}} | 0 \rangle \right|^2 \delta\left(\omega - (E_n - E_0)\right)$$

This can be calculated within `Su(n)ny` using the continued fraction method and we can then look at when $\vec{q}=0$. However this procedure fails when looking at higher order
responses.

For second order response we must perform an internal summation over all eigenstates to obtain the Non-linear response:

$$ 
\chi^{(2)}_{\alpha \beta \gamma}(t_2, t_1) = -\frac{1}{N}\sum_{PQ}\left[{{m^\alpha_{0Q}m^\beta_{QP}m^\gamma_{P0}}\cos\left(E_Pt_1+E_Qt_2\right)+ {m^\alpha_{0Q}m^\beta_{QP}m^\gamma_{P0}}
\cos\left(E_Pt_1+
\Delta E_{PQ}t_2\right)}\right]
$$

where $m^\alpha_{AB} = \langle A | M^\alpha | B \rangle$. This summtion can be done using continued fraction however it gets messy and quite difficult.
This repositry should hold examples of the adjusted sunny branch.
