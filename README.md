# Automatic Routing of Customer Support Messages

Project for AI for Business. A machine-learning model reads a customer
message and predicts its topic, so it can go to the right team or chatbot.

Full report: `report/Project1_report.docx`

## Goal
Sorting support messages by hand is slow and causes mistakes. Customers wait
longer, agents lose time on sorting, and the business pays more. This project
compares five scikit-learn models that classify messages into 11 categories,
so messages can be routed automatically.

## Data
- File: `data/customer_support.csv` (2,200 messages, 200 per category)
- Columns: `message`, `category`
- Categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER,
  PAYMENT, REFUND, SHIPPING, SUBSCRIPTION
- No missing values and no duplicates. Messages have about 9 words on average,
  and 13% contain placeholders such as `{{Order Number}}`.
- Source: Bitext Customer Support dataset (CDLA-Sharing-1.0 licence)

**Credit:** Bitext Innovations. (2024). *Bitext Customer Service Tagged
Training Dataset for LLM-based Virtual Assistants* [Dataset]. Hugging Face.
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

## Method
1. Explore the data (size, balance, duplicates, message length).
2. Stratified 80/20 train/test split (`random_state=42`).
3. TF-IDF features inside a scikit-learn pipeline (no data leakage).
4. Models: Dummy baseline, MultinomialNB, Decision Tree, Random Forest, KNN (k=5).
5. Evaluation: accuracy, weighted F1, 10-fold cross-validation, confusion matrices.
6. Extras: top words per category, typo robustness test, confidence threshold.

## Results (test set, 440 messages)
| Model | Accuracy | Weighted F1 | 10-fold CV |
|---|---|---|---|
| Dummy (baseline) | 0.091 | 0.015 | 0.091 |
| MultinomialNB | 0.995 | 0.995 | 0.986 |
| Decision Tree | 0.941 | 0.942 | 0.946 |
| Random Forest | 0.986 | 0.986 | 0.982 |
| KNN (k=5) | 0.984 | 0.984 | 0.975 |

Best model: Multinomial Naive Bayes (highest scores, fast, gives probabilities).
The Dummy model always predicts ACCOUNT, so it has no useful confusion pair.

![Naive Bayes confusion matrix](outputs/cm_MultinomialNB.jpg)

## Key findings
- All four models are far above the 9.1% baseline (chance level for 11 classes).
- Multinomial Naive Bayes is best (99.5% accuracy, 2 errors out of 440).
- The Decision Tree is weakest (94.1%); errors are mostly CONTACT/ORDER.
- Scores are very high because the data is synthetic and template-like.
  Real-world accuracy will likely be lower.

## Robustness and threshold tests
Typo test (random letter swaps in the test messages, accuracy):

| Model | Clean | With typos |
|---|---|---|
| MultinomialNB | 0.995 | 0.898 |
| Decision Tree | 0.941 | 0.675 |
| Random Forest | 0.986 | 0.811 |
| KNN (k=5) | 0.984 | 0.836 |

Confidence threshold (Naive Bayes):

| Threshold | Auto-routed | Accuracy on those |
|---|---|---|
| 0.3 | 99% | 0.998 |
| 0.5 | 91% | 1.000 |
| 0.7 | 71% | 1.000 |

## Business use
Auto-route only messages with high model confidence (for example 0.5 or
higher); send the rest to a human agent. Start with a pilot where agents
confirm the suggestion.

## Responsible AI
- Privacy: mask personal data before training (GDPR).
- Fairness: test with typos and non-native writing; retrain with noisy examples.
- Human control: customers can always reach an agent.

## Repository structure
├── data/customer_support.csv
├── notebooks/project1.ipynb
├── outputs/
│ ├── results.csv
│ ├── top_words.csv
│ └── cm_*.jpg confusion matrices
├── report/Project1_report.docx
├── requirements.txt
└── README.md

## How to run
1. Clone the repo and open the folder in VS Code.
2. Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate # Mac/Linux: source .venv/bin/activate
3. Install packages: `pip install -r requirements.txt`
4. Open `notebooks/project1.ipynb`, select the `.venv` kernel and click **Run All**.
5. Results and figures are saved in `outputs/`.

Requires Python 3.10+. Packages: pandas, scikit-learn, matplotlib, seaborn, ipykernel.

## Limitations
- The data is synthetic, balanced and English only. Real accuracy will likely be lower.
- Some top words are artefacts (for example invoice numbers).
- Each message gets one label, so messages with two topics are not handled.
- The typo and threshold tests use the same test set, so they show relative
differences, not exact real-world accuracy.

## Report and figures
- Full report: `report/Project1_report.docx`
- Confusion matrices: `outputs/cm_*.jpg`

## Licence
Code: MIT (see `LICENSE`). Data: CDLA-Sharing-1.0 (Bitext).

## Author
Kamila Dionizy, AI for Business, 21.11.2026
