### Code Disclosure
This Repository is a fork of https://github.com/junghunkim7786/contextual_linear_bandits_under_noisy_features
from the paper "Contextual Linear Bandits under Noisy Features: Towards Bayesian Oracles" [1]

## Running the Repository
To install poetry
```
python -m pip install poetry
python -m poetry init
python -m poetry install
```
This will generate a poetry.lock file and install the dependencies from pyproject.toml
To run a file using the poetry environemnt:
```
python -m poetry run python scheduler_wine.py
```

or run the poetry shell and call python directly
```
python -m poetry shell
python scheduler_wine.py
```

**Code Flow**

Scheduler -> Preprocess Data -> Train Auto Encoder -> Load Environment -> Run CMAB Algorithms -> Plot Results

**Python File execution**

scheduler_wine.py -> wine_preprocess.py -> wine_aetrain.py -> env_wine.py -> real_main.py -> real_plotting.py

### References

[1] Kim, J., Yun, S., Jeong, M., Nam, J., Shin, J. &amp; Combes, R.. (2023). Contextual Linear Bandits under Noisy Features: Towards Bayesian Oracles. <i>Proceedings of The 26th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 206:1624-1645 Available from https://proceedings.mlr.press/v206/kim23b.html.