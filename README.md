# ant colony algorithm (ant system implementation)

Reimplementation of Marco Dorigo's Ant System optimization algorithm using Python. Applicable to shortest-path problem and traveling salesman problem. 

## Installation

### Local installation using Python virutalenv and Makefile:

1. run `make venv`
2. run `source venv/Scripts/activate`

## Usage

- Run tests using `make test`
- Run example program using `make example`
- Run example using cli commands (see parameter definitions in `example.py`):
```
python example.py -k 2 -c 1 -v 9 -pr 'shortest_path' -s 19
```

