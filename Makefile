test:
	python -m pytest --tb=short tests.py

watch-tests:
	ls *.py | entr python -m pytest --tb=short

black:
	black -l 86 -t py310 .