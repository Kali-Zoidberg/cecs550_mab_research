To install poetry
```
pip install poetry
poetry init
poetry install
```
This will generate a poetry.lock file and install the dependencies from pyproject.toml
To run a file using the poetry environemnt:
```
python -m poetry run "file.py"
```

or run the shell and call python directrly
```
poetry shell
python "file.py"
```

To add dependencies
```
poetry add "dependency"
```
This will add the dependency to the pyproject.toml\
Note, "dependency" is the same name you would use as if install the dependency using pip.
