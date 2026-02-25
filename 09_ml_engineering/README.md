# Module 09: ML Engineering

Production ML patterns using pure Python (no numpy/sklearn required).

## Topics Covered

| File | Concepts |
|------|----------|
| `pipeline/feature_engineering.py` | Scaling, encoding, feature selection |
| `pipeline/model_trainer.py` | Training loop, cross-validation |
| `pipeline/model_evaluator.py` | Accuracy, precision, recall, F1, confusion matrix |
| `serving/inference_api.py` | FastAPI model serving (conditional import) |

## Running

```bash
python pipeline/model_trainer.py   # trains a logistic regression from scratch
pytest tests/ -v
```
