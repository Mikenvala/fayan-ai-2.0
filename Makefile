.PHONY: install run test clean docker

install:
	pip install -r internship-tasks/task4-unified-platform/requirements.txt
	pip install markdown

run:
	cd internship-tasks/task4-unified-platform && python -m uvicorn backend:app --host 0.0.0.0 --port 8800

test:
	python -m py_compile internship-tasks/task4-unified-platform/backend.py
	python -m py_compile internship-tasks/task4-unified-platform/multi_agent.py

docker:
	docker-compose up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
