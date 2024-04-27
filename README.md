<p align="center">
  <img aligne="center" src="images/CS.png" width="45%" />
  <img aligne="center" src="images/artefact.png" width="50%" ver />
</p>

# Decision Systems and Preferences
Auriau Vincent,
Belahcene Khaled,
Mousseau Vincent

## Table of Contents
- [Decision Systems and Preferences](#decision-systems-and-preferences)
- [Repository Usage](#repository-usage)
- [Context](#context)
- [Taks](#tasks)
- [Deliverables](#deliverables)
- [Resources](#resources)

## Repository usage
1.  Install [git-lfs ](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), it will be needed to download the data
2. Fork the repository, the fork will be used as a deliverable
3. Clone your fork and push your solution model on it

The command 
```bash
conda env create -f config/env.yml
conda activate cs_td
python evaluation.py
``````
will be used for evaluation, with two other test datasets. Make sure that it works well.

## Context
Our main objective is to better understanding customer preferences through their purchases.
We see a customer as a decision function when he comes to the supermarket. Once facing the shelf of a type of product he wants to buy, the customer assesses what are the different alternatives available. Considering the size, price, brand, packaging, and any other information, the customer ranks all the products in his mind and finally chooses his preferred alternative.
We have at our disposal a list of P expressed preferences. These preferences illustrate that a customer has preferred - or chosen - X[i] over Y[i]. We split the task of determining customers preferences into two sub-tasks:
    -  We want to clusterize the customers through their purchases so that customers with similar decisions are grouped together.
    - We want to determine for each cluster the decision function that lets the customers rank all the products.
## Modeling

### Classical Modeling - 

This comprehensive overview combines insights into the decision function's role in modeling customer preferences within a market segmentation framework, grounded in utility-based choice theory. The model employs mixed integer programming (MIP) to navigate the discrete and nonlinear nature of customer choices and preferences.

## Decision Function: Quantifying Customer Preferences

The decision function quantifies the preference of a customer cluster \(k\) for a product pair \(j\), playing a crucial role in understanding the influence of product attributes on customer choice. It distinguishes between the utility of purchased products \(x\) and non-chosen, replaceable products \(y\).

### For the Purchased Product $x$:

The utility of product $x$ for customer cluster $k$ and product pair $j$ is given by:

$$
u_k(x^{(j)}) = \sum_{i=1}^{n} u_{k,i}(x_i^{(j)}) + \sigma^{-}(x_i^{(j)}) - \sigma^{+}(x_i^{(j)})
$$

Where:

- $u_k(x^{(j)})$: Total utility of product \(x\) for customer cluster \(k\) and product pair \(j\).
- $u_{k,i}(x_i^{(j)})$: Partial utility from attribute \(i\) of product \(x\) for cluster \(k\).
- $\sigma^{-}(x_i^{(j)})$, $\sigma^{+}(x_i^{(j)})$: Slack variables for underestimation and overestimation of utility, respectively.

### For Non-Chosen Replaceable Products $y$:

Similarly, the utility for non-chosen products follows the same structure but applies to the alternatives that were not selected:

$$
u_k(y^{(j)}) = \sum_{i=1}^{n} u_{k,i}(y_i^{(j)}) + \sigma^{-}(y_i^{(j)}) - \sigma^{+}(y_i^{(j)})
$$

- \(u_k(y^{(j)})\): Total utility of product \(y\) for customer cluster \(k\) and product pair \(j\).

## Decision (Indicator) Variable:

This binary variable indicates the preference of cluster \(k\) between the purchased and non-chosen products for each pair \(j\):

$$
z_{j,k} = 
\begin{cases}
0, & \text{if } u_k(x^{(j)}) < u_k(y^{(j)}) \\
1, & \text{if } u_k(x^{(j)}) \geq u_k(y^{(j)})
\end{cases}
$$

## Objective Function:

The objective is to minimize the total slack, ensuring the model's utility estimations are as accurate as possible without significant under or overestimation:

$$
\min \sum_{j=1}^{P} (\sigma^{-}(x^{(j)}) + \sigma^{+}(x^{(j)}) + \sigma^{-}(y^{(j)}) + \sigma^{+}(y^{(j)}))
$$

## Constraints for Mixed Integer Programming (MIP):

### Customer Preference Constraint:

Ensures at least one customer cluster prefers $x$ over $y$ for each pair $j$:

$$
\sum_{k=1}^{K} z_{j,k} \geq 1
$$

### Indicator Constraint:

A large \(M\) facilitates the modeling of the decision variable, linking it with the utility differences:

$$
M(1-z_{j,k}) \leq (u_k(x^{(j)}) - u_k(y^{(j)})) < M \cdot z_{j,k}
$$

### Monotonicity Constraint:

Guarantees the utility increases with attribute levels, ensuring model consistency and interpretability:

$$
u_{k,i}(x_i^{(j)})^{l+1} - u_{k,i}(x_i^{(j)})^{l} \geq \epsilon, \forall i, \forall k, l=0...L_i-1
$$

### Normalization Constraint:

Normalizes the utility scale and sets a reference point, aiding in utility comparison across attributes:

$$
u_{k,i}(x_i^{(j)})^{0} = 0, \forall i, \forall k
$$

$$
\sum u_{k,i}(x_i^{(j)})^{L_i} = 1
$$

## Loss Function
$$
    L_{\text{total}} =  \lambda_{neg} L_{\text{neg}} + \lambda_p L_{\text{pref}} + \lambda_{\text{min}}L_{\text{min}} + \lambda_{\text{max}}L_{\text{max}} + P_{\text{neg}} - \sum_i \log((\Delta U ^ {(i)})^2 + \varepsilon)
$$

Where:

$$
\lambda_{neg} = 100 \quad \lambda_{p} = 0.1 \quad \lambda_{\text{min}}=\lambda_{\text{max}}=10 
$$

$$
\Delta U ^ {(i)} = U_x^{(i)} - U_y^{(i)}
$$


$$
L_{\text{pref}} = \frac{1}{|X|}\sum_{x,y \in X,Y}\left(\prod_i\text{ReLU}(\Delta U ^ {(i)})\right)
$$

$$
L_{\text{neg}} = \sum_{x,y \in X,Y}\Bigg(\Big(\sum_i\mathcal{H}(-\Delta U^ {(i)})  \Big) - 1\Bigg)^2
$$

$$
L_{\text{min}} = \left((U_{\text{min}} - 0)^2\right)
$$

$$
L_{\text{max}} = \left((U_{\text{max}} - 1)^2\right)
$$

$$
P_{\text{neg}} = -\sum_{i} \mathbf{1}_{\sum_j \mathcal{H}(-\Delta U^{(j)}) > 0} \log((\Delta U ^ {(i)})^2 + \varepsilon)
$$




## Tasks
You are asked to:
  - Write a Mixed-Integer Progamming model that would solve both the clustering and learning of a UTA model on each cluster
  - Code this MIP inside the TwoClusterMIP class in python/model.py. It should work on the dataset_4 dataset.
  - Explain and code a heuristic model that can work on the dataset_10 dataset. It should be done inside the HeuristicModel class.

## Deliverables
You will present your results during an oral presentation organized the on Tuesday $13^{th}$ (from 1.30 pm) of February. Exact time will be communicated later. Along the presentation, we are waiting for:

-  A report summarizing you results as well as your thought process or even non-working models if you consider it to be interesting.
-  Your solution of the first assignement should be clearly written in this report. For clarity, you should clearly state variables, constraints and objective of the MIP.
-  A well organized git repository with all the Python code of the presented results. A GitHub fork of the repository is preferred. Add some documentation directly in the code or in the report for better understanding. The code must be easily run for testing purposes.
- In particular the repository should contain your solutions in the class TwoClustersMIP and HeuristicModel in the models.pu file.  If you use additional libraries, add them inside the config/env.ymlfile. The command 'python evaluation.py' will be used to check your models, be sure that it works and that your code complies with it. The dataset used will be a new one, with the same standards as 'dataset\_4' and 'dataset\_10'.

## Resources
- [Gurobi](https://www.gurobi.com/)
- [Example Jupyter Notebook](notebooks/example.ipynb)
- [UTA model](https://www.sciencedirect.com/science/article/abs/pii/0377221782901552)
