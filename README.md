# In-depth Analysis of Low-rank Matrix Factorisation in a Federated Setting
We present here the code of the experimental parts of the following paper:
```
C. Philippenko, K. Scaman and L. Massoulié, In-depth Analysis of Low-rank Matrix Factorisation in a Federated Setting 
least-squares regression: application to Federated Learning, 2024.
```

In this paper, we analyze a distributed algorithm to compute a low-rank matrix factorization on $N$ clients, each 
holding a local dataset $\mathbf{S}^i \in \mathbb{R}^{n_i \times d}$, mathematically, we seek to solve 
$min_{\mathbf{U}^i \in \mathbb{R}^{n_i\times r}, \mathbf{V}\in \mathbb{R}^{d \times r} } \frac{1}{2} \sum_{i=1}^N \|\mathbf{S}^i - \mathbf{U}^i \mathbf{V}^\top\|^2_{\text{F}}$. 
Considering a power initialization of $\mathbf{V}$, we rewrite the previous smooth non-convex problem into a smooth 
strongly-convex problem that we solve using a parallel nesterov gradient descent potentially requiring a single step of 
communication at the initialization step. For any client $i$ in $\{1, \dots, N\}$, we obtain a global $\mathbf{V}$ in 
$\mathbb{R}^{d \times r}$ common to all clients and a local variable $\mathbf{U}^i$ in $\mathbb{R}^{n_i \times r}$. We 
provide a linear rate of convergence of the excess loss which depends on $\kappa^{-1}(\mathbf{S})$, where $\mathbf{S}$ 
is the concatenation of the matrix $(\mathbf{S}^i)_{i=1}^N$ and $\kappa(\mathbf{S})$ its condition number. This result 
improves the rates of convergence given in the literature, which depend on $\kappa^{-2}(\mathbf{S})$. We provide an 
upper bound on the Frobenius-norm error of reconstruction under the power initialization strategy. We complete our 
analysis with experiments on both synthetic and real data.


From our analysis, several take-aways can be identified.
1. Increasing the number of communication $\alpha$ leads to reduce the error $\epsilon$ by a factor 
$\sigma_{r_\*+1}^{4\alpha}/ \sigma_{r_\*}^{4\alpha}$, therefore, getting closer to the minimal Frobenius-norm error 
$\epsilon$.
2. Using a gradient descent instead of an SVD to approximate the exact solution of the strongly-convex problem allows us 
to bridge two parallel lines of research. Further, we obtain a simple and elegant proof of convergence and all the 
theory from optimization can be plugged in.
3. By sampling several Gaussian matrix $\Phi$, we improve the rate of convergence of the gradient descent. Further, 
based on random Gaussian matrix theory, it results in an almost surely convergence if we sample $\Phi$ until 
$\mathbf{V}$ is well conditioned.
4. 
## Running experiments

Run the following commands to generate the illustrative figures in the article.

```python3 -m src.plotter_script.PlotRealDataset --dataset_name synth```

```python3 -m src.plotter_script.PlotRealDataset --dataset_name mnist```

```python3 -m src.plotter_script.PlotRealDataset --dataset_name celeba```

```python3 -m src.plotter_script.PlotRealDataset --dataset_name w8a```

## Requirements

Using pip:
```pip install -c conda-forge -r requirements.txt python=3.7```. 

Or to create a conda environment: ```conda create -c conda-forge --name matrix_factorisation --file requirements.txt python=3.7```.

## Maintainers

[@ConstantinPhilippenko](https://github.com/philipco)

## License

[MIT](LICENSE) © Constantin Philippenko

# References
If you use this code, please cite the following papers

```
@article{philippenko2024indepth,
  title={In-depth Analysis of Low-rank Matrix Factorisation in a Federated Setting},
  author={Philippenko, Constantin and Scaman, Kevin and Massoulié, Laurent},
  journal={arXiv e-prints},
  year={2024}
}
```