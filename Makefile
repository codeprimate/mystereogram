.PHONY: clean build check publish install install-dev

# ==============================================================================
# Build and Publishing
# ==============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ .eggs/ *.egg-info

build:
	@echo "🔨 Building package..."
	python -m build

check:
	@echo "✅ Checking distribution files..."
	twine check dist/*

publish: clean build check
	@echo "🚀 Publishing to PyPI..."
	twine upload dist/*

# ==============================================================================
# Local Installation
# ==============================================================================

install: build
	@echo "📦 Installing package locally..."
	pip install dist/*.whl

install-dev:
	@echo "📦 Installing package in editable mode (for development)..."
	pip install -e .

