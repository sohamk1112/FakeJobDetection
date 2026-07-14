# FakeJobDetection
**Problem:** Job postings on online platforms are vulnerable to scams that harvest personal data or charge bogus fees from applicants.

**Solution:** End to end machine learning pipeline to classify job postings as legitimate or fraudulent.

## Dataset
[Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (EMSCAD, University of the Aegean)
- 17,880 job postings
- 95% legitimate, 5% fraudulent (severe class imbalance)

## Approach
1. **EDA:** Identified metadata signals (missing company profile = 9x fraud rate)
2. **Preprocessing:** TFIDF vectorisation + one hot encoding + stratified 70/15/15 split
3. **Models:** Trained 6 classifiers (Logistic Regression, Naive Bayes, Linear SVM, Random Forest, Decision Tree, Voting Ensemble)
4. **Evaluation:** F1score (0.8327 on test set), precision recall analysis, error analysis
5. **Reflection:** Ethical considerations, interpretability trade-offs, limitations

## Results
- **Test F1 (fraud class):** 0.8327
- **Precision:** 0.8425
- **Recall:** 0.8231
- Catches ~82% of scams with low false alarm rate

## Files
- `25201173_FakeJobDetection.ipynb` - Full Jupyter notebook with code, outputs, and analysis
- `requirements.txt` - Python dependencies

## Usage
```bash
pip install -r requirements.txt
jupyter notebook 25201173_FakeJobDetection.ipynb
```

## Key Insights
- Class imbalance requires F1 score, not accuracy
- Missing metadata (company profile, logo) is the strongest fraud signal
- Voting ensemble outperforms individually tuned linear models
- Model struggles with sophisticated, well-written fraud

## Limitations & Future Work
- TF-IDF cannot capture semantic deception
- Dataset from 2014, modern AI generated scams may bypass detection
- Future: fin -tuned transformers (BERT), explainability (SHAP), fresh 2025 data
