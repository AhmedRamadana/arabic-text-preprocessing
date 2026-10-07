# Arabic Text Preprocessing & Spell Correction

A notebook that cleans and preprocesses raw Arabic tweets (single `text` column only — no labels, no sentiment analysis) and **compares three spell-correction approaches** side by side: a rule-based dictionary, **SWEET** (CAMeL-Lab), and an **AraT5-based** seq2seq model.

📓 **Notebook:** [`Task_1_Arabic_Text_Preprocessing.ipynb`](Task_1_Arabic_Text_Preprocessing.ipynb)

---

## Dataset

[Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus) (Kaggle, `mksaad`), downloaded with `kagglehub`.

The positive and negative train/test files are concatenated, then the `label` column is dropped immediately so only `text` remains.

---

## Pipeline

```
Raw Twitter Text (train / test)
        │
        ▼
1. Load data — drop `label`, keep `text` only
2. Language diagnostics (English / Franco / Arabic counts)
3. Regex cleaning — URLs, mentions, HTML tags, tatweel
4. Remove punctuation (Arabic + English)
5. Remove numbers (Arabic + English digits)
6. Remove emojis
7. Franco → Arabic dictionary translation + drop remaining English
8. Arabic normalization (أ/إ/آ → ا, ى → ي, ة → ه, ؤ → و, ئ → ي)
9. Spell correction — 3 methods compared
10. Tokenization            ┐
11. Arabic stopword removal ├ planned (see "Status" below)
12. Stemming (ISRIStemmer)  ┘
        │
        ▼
   `clean_text` — ready for downstream tasks
```

### Why this order?

- **Regex cleaning first**, otherwise URLs and mentions get split into meaningless tokens.
- **Normalization before spell correction**, so the correction models react only to *real* spelling mistakes and not to character-variant differences.
- **Spell correction before tokenization / stopwords / stemming**, because the models need full, uncut sentences. Correcting after stemming would mean correcting already-reduced word roots.

---

## Spell correction: three methods compared

| | Dictionary | SWEET | AraT5-based |
|---|---|---|---|
| **Model** | None (rule-based) | `CAMeL-Lab/text-editing-coda` | `NAMAA-Space/SaudiSpell-AraT5` |
| **Type** | Fixed mistake → correction map | BERT token classification (edit tags), then `gec.tag.rewrite()` | T5 seq2seq (generates the corrected sentence) |
| **Trained on** | — | MADAR CODA (dialectal Arabic GEC) | Saudi dialect + MSA spelling corpus |
| **Inference cost** | Instant | One forward pass | Autoregressive generation with beam search (slowest) |
| **Strength** | Deterministic, zero dependencies | Fast, good fit for dialectal Twitter text | Context-aware, free-form corrections |

All three methods are run on the same sample sentences, and the notebook produces a comparison table with the output of each method plus per-sentence inference time.

SWEET is chosen for the full dataset because of its speed. AraT5 remains the higher-context alternative when a GPU is available.

---

## Getting started

The notebook was written for **Google Colab** (paths use `/content/...`).

1. Open the notebook in Colab (or Jupyter).
2. Make sure you have access to Kaggle datasets through `kagglehub`.
3. Run the cells in order. Dependencies are installed inside the notebook:

```bash
pip install kagglehub transformers torch sentencepiece camel-tools editdistance
git clone https://github.com/CAMeL-Lab/text-editing.git   # provides gec.tag.rewrite
```

### Main libraries

`pandas`, `numpy`, `re`, `kagglehub`, `transformers`, `torch`, `sentencepiece`, `camel-tools`, `editdistance`

---

## Status

- [x] Data loading (text only)
- [x] Language diagnostics
- [x] Regex, punctuation, number and emoji cleaning
- [x] Franco → Arabic translation (small demo dictionary)
- [x] Arabic normalization
- [x] Spell correction: Dictionary vs. SWEET vs. AraT5 (timed comparison)
- [ ] Tokenization, Arabic stopword removal, ISRIStemmer stemming
- [ ] Applying the chosen spell-correction model to the full train/test sets and saving `clean_text`

---

## Notes and limitations

- The Franco dictionary and the spell-correction dictionary are **small samples** meant to demonstrate the approach, not complete resources.
- Any English word not found in the Franco dictionary is dropped.
- Both transformer models are downloaded from the Hugging Face Hub on first run.

---

## Acknowledgements

- [CAMeL-Lab/text-editing](https://github.com/CAMeL-Lab/text-editing) (SWEET)
- [NAMAA-Space/SaudiSpell-AraT5](https://huggingface.co/NAMAA-Space/SaudiSpell-AraT5)
- [Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus) by mksaad