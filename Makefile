.PHONY: install run data features train test clean help

# Default Python interpreter
PYTHON := python3

# Directories
DATA_DIR := .
MODELS_DIR := models
PLOTS_DIR := plots
RESULTS_DIR := results

# Files
RAW_DATA := test4.csv
PROCESSED_DATA := processedtest4_enhanced.csv
REQUIREMENTS := requirements.txt

help:
	@echo "Available targets:"
	@echo "  make install    - Install all Python dependencies"
	@echo "  make data       - Download stock data (AAPL)"
	@echo "  make features   - Extract features from raw data"
	@echo "  make train      - Train all models"
	@echo "  make test       - Run test suite"
	@echo "  make run        - Run complete pipeline (data + features + train)"
	@echo "  make clean      - Clean generated files"
	@echo "  make help       - Show this help message"

install:
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "✓ Dependencies installed"

data:
	@echo "Downloading stock data..."
	$(PYTHON) getData.py
	@echo "✓ Stock data downloaded"

features:
	@echo "Extracting features..."
	@if [ ! -f $(RAW_DATA) ]; then \
		echo "Error: $(RAW_DATA) not found. Run 'make data' first."; \
		exit 1; \
	fi
	$(PYTHON) feature_extraction_enhanced.py
	@echo "✓ Features extracted"

train:
	@echo "Training models..."
	@if [ ! -f $(PROCESSED_DATA) ]; then \
		echo "Error: $(PROCESSED_DATA) not found. Run 'make features' first."; \
		exit 1; \
	fi
	@echo "Note: Full training is done via Jupyter notebook (s3.4testsentiment2.ipynb)"
	@echo "For command-line training, use: python model-train.py --model logistic"
	@echo "✓ Training instructions displayed"

test:
	@echo "Running tests..."
	$(PYTHON) test_project.py
	@echo "✓ Tests completed"

run: install data features
	@echo "✓ Complete pipeline executed"
	@echo "Next steps:"
	@echo "  1. Open s3.4testsentiment2.ipynb in Jupyter"
	@echo "  2. Run all cells to train models and generate results"

clean:
	@echo "Cleaning generated files..."
	rm -rf $(MODELS_DIR)/*.pkl
	rm -rf $(PLOTS_DIR)/*.png
	rm -rf $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.txt
	rm -f $(PROCESSED_DATA) processedtest4*.csv
	@echo "✓ Cleaned generated files"

clean-all: clean
	@echo "Cleaning all data files..."
	rm -f $(RAW_DATA) test4.csv
	rm -f apple_*.csv
	@echo "✓ Cleaned all data files"
