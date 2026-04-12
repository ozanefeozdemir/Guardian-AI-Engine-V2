.PHONY: validate preprocess train-baseline train-contrastive train-federated adapt evaluate test clean

validate:
	python scripts/validate_data.py

preprocess:
	python scripts/preprocess.py --config configs/base.yaml

train-baseline:
	python scripts/train_baseline.py --config configs/phase1_baseline.yaml

train-contrastive:
	python scripts/train_contrastive.py --config configs/phase2_contrastive.yaml

train-federated:
	python scripts/train_federated.py --config configs/phase3_federated.yaml

adapt:
	python scripts/adapt_fewshot.py --config configs/phase4_fewshot.yaml

evaluate:
	python scripts/evaluate.py --protocol b

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
