# Makefile for Agentic Observability Platform

.PHONY: help install install-dev test lint format clean docker-up docker-down run

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-logs   - View Docker logs"
	@echo "  make run           - Run API server (local dev)"
	@echo "  make run-docker    - Run full stack in Docker"
	@echo "  make test-api      - Test API endpoints"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src/agentic_observability --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	sleep 5
	docker-compose ps

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f app

run-docker:
	docker-compose up

test-api:
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/api/v1/health | python -m json.tool
	@echo "\nTesting detailed health..."
	curl -s http://localhost:8000/api/v1/health/ready | python -m json.tool
	@echo "\nTesting metrics endpoint..."
	curl -s http://localhost:8000/api/v1/metrics | python -m json.tool

run:
	uvicorn agentic_observability.api.main:app --app-dir src --reload --port 8000

run-prod:
	uvicorn agentic_observability.api.main:app --app-dir src --host 0.0.0.0 --port 8000
