# Non-IID Movie Recommendation

This repository contains the code and resources related to the project developed for the **Recommendation Systems (BCC409)** course. The project aims to explore **Federated Learning** in scenarios with **non-IID data** (non-Independent and Identically Distributed) for the task of movie recommendation.

![python](https://img.shields.io/badge/python-3.11.0-f7ca54?style=for-the-badge&logo=python)
![issues](https://img.shields.io/github/issues/arthurnfmc/RecomendacaoDeFilmesNaoIID?style=for-the-badge)
![forks](https://img.shields.io/github/forks/arthurnfmc/RecomendacaoDeFilmesNaoIID?style=for-the-badge)
![stars](https://img.shields.io/github/stars/arthurnfmc/RecomendacaoDeFilmesNaoIID?style=for-the-badge)
![license](https://img.shields.io/github/license/arthurnfmc/RecomendacaoDeFilmesNaoIID?style=for-the-badge)

## Table of Contents

- [Non-IID Movie Recommendation](#non-iid-movie-recommendation)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
    - [Objectives](#objectives)
  - [Technologies](#technologies)
  - [Installation](#installation)
  - [Contributors](#contributors)
  - [License](#license)

## Description

The project focuses on applying **Federated Learning** techniques in Recommendation Systems (RS), especially in scenarios where data is distributed in a non-IID manner. The main motivation is to evaluate how non-IID data distribution affects the performance of recommendation models and to explore strategies to mitigate these effects, such as partial data sharing among network nodes.

### Objectives

1. **Evaluate the impact of non-IID data** on the performance of recommendation models.
2. **Implement and test the data sharing strategy** proposed by Zhao et al [^1] to mitigate the negative effects of non-IID data.
   [^1]: Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., Amodei, D., & Smola, A. (2018). Federated learning with non-iid data. arXiv preprint arXiv:1806.00582.
3. **Compare the performance** of models trained in IID and non-IID scenarios.

## Technologies

The project was developed using the Python programming language and the following libraries:

- python==3.11.0
- pandas==2.2.3
- numpy==2.2.2
- scikit-learn==1.6.1
- tensorflow==2.6.0
- datasets==3.1.1
- flwr==1.15.1

Jupyter Notebook was used for running the experiments.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/arthurnfmc/RecomendacaoDeFilmesNaoIID.git
    cd RecomendacaoDeFilmesNaoIID
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Contributors

- Arthur Negrão
- Guilherme Rocha
- Lucas Gomes
- Pedro Igor

## License
