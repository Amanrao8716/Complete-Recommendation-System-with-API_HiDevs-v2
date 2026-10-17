.PHONY: dashboard install lint format typecheck test coverage check seed run docker-build docker-run

install:
	pip install -r requirements-dev.txt

lint:
	flake8 .
	black --check .

format:
	black .

typecheck:
	mypy

test:
	pytest

coverage:
	pytest --cov --cov-fail-under=85 --cov-report=term-missing

check: lint typecheck coverage

seed:
	python scripts/seed_data.py --reset

run:
	python -m api.app

docker-build:
	docker build -t recsys .

docker-run:
	docker run --rm -p 5000:5000 -v recsys-data:/data recsys

dashboard:
	pip install -r requirements-ui.txt
	streamlit run dashboard/streamlit_app.py
