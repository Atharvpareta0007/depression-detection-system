.PHONY: help install test run deploy clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make run         - Run development server"
	@echo "  make deploy      - Deploy to production"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up   - Start Docker containers"
	@echo "  make docker-down - Stop Docker containers"
	@echo "  make clean       - Clean temporary files"

install:
	pip install -r backend/requirements.txt
	pip install -r requirements.txt
	cd frontend && npm install

test:
	pip install -r requirements-dev.txt
	pytest tests/ -v

run:
	cd backend && python app.py

deploy:
	./deploy.sh production

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf frontend/dist
	rm -rf frontend/node_modules/.cache

