all: venv test example

venv:
	virtualenv venv\
	&& source venv/Scripts/activate\
	&& python -m pip install --upgrade pip\
	&& python -m pip install -r requirements.txt

test:
	python -m pytest --tb=short tests.py

example:
	python example.py -k 2 -c 1 -v 9 -pr 'shortest_path' -s 19

watch-tests:
	ls *.py | entr python -m pytest --tb=short tests.py

black:
	black -l 120 -t py310 .