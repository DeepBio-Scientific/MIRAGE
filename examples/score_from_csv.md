# Scoring from a CSV (any language, any model)

If your model is not in Python, or you have already generated predictions, you do not
need to touch the Python API. Produce a two-column CSV and score it.

## 1. Get the inputs

```python
import mirage
df = mirage.load("redundancy", core=True)   # id, sequence, smiles, pK, family_size, ...
df[["id", "sequence", "smiles"]].to_csv("inputs.csv", index=False)
```

Or download `redundancy.parquet` / `temporal.parquet` directly from the Hugging Face
dataset and read them with any parquet reader.

## 2. Predict with your model

Score every `(sequence, smiles)` pair and write:

```csv
id,prediction
1a4k,6.31
1abc,4.02
...
```

`prediction` is any score where **higher = stronger binder**. If yours is the opposite
(e.g. log10(IC50)), add `--lower-is-stronger` when scoring.

## 3. Score

```bash
mirage score redundancy preds.csv
mirage score temporal   preds_temporal.csv --lower-is-stronger
```

You will get the family-size dose-response, the novel-family headline number, and the
comparison against molecular weight.
