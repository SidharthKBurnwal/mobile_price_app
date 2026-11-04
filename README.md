# Phone Price-Band Lab: Mobile Price Classification

ML Assignment 2 (M.Tech AIML, Machine Learning). Five classifiers, one dataset, one Streamlit app.

## a. Problem statement

Given the hardware specification of a mobile phone (battery, RAM, screen resolution,
camera, connectivity, and so on), predict which **price band** it belongs to:
`0 = Low`, `1 = Medium`, `2 = High`, `3 = Very high`. This is a **4-class classification**
problem. The goal is to compare five classic classifiers on the same hold-out data and
expose them through an interactive web app.

## b. Dataset description

| Item | Value |
|---|---|
| Source | Kaggle, *Mobile Price Classification* (`train.csv`) - https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification |
| Instances | 2000 |
| Features | 20 (battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi) |
| Target | `price_range` (0-3), 500 samples per class (balanced) |
| Missing values | None |
| Split | Stratified 80 / 20, `random_state = 2026` -> 1600 train / 400 test |
| Preprocessing | `StandardScaler` inside each pipeline for Logistic Regression, KNN, Naive Bayes and SVM (not used for the Decision Tree). Fitted on the training split only and saved together with the model. |
| Tuning | 5-fold stratified `GridSearchCV` (scoring = accuracy) on the training split only |

`test_data.csv` is the 20 % hold-out split (features + target). All numbers below are
computed on it.

## c. GitHub repository link

`https://github.com/SidharthKBurnwal/mobile_price_app/tree/main`

Live app: `https://mobilepriceapp-mnrnq7pceh3nxjccq3yiwe.streamlit.app/`

Repository layout:

```
app.py              Streamlit app (loads saved models, never re-trains)
common.py           shared constants + metric helpers
train_models.py     training / evaluation script (writes model/ and test_data.csv)
requirements.txt
test_data.csv       hold-out test data (features + target)
data/train.csv      original Kaggle file
model/              logistic_regression / decision_tree / knn / naive_bayes / svm .joblib, metrics.csv
```

## d. Models used

Precision, Recall and F1 are **weighted averages** over the 4 classes. AUC is
**one-vs-rest, weighted**. MCC is the multi-class Matthews coefficient.

Results on the 400-row hold-out set (`python train_models.py`, also saved in `model/metrics.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9650 | 0.9982 | 0.9649 | 0.9650 | 0.9649 | 0.9533 |
| Decision Tree | 0.8575 | 0.9452 | 0.8582 | 0.8575 | 0.8576 | 0.8102 |
| KNN | 0.6075 | 0.8277 | 0.6124 | 0.6075 | 0.6098 | 0.4768 |
| Naive Bayes (Gaussian) | 0.8075 | 0.9449 | 0.8144 | 0.8075 | 0.8093 | 0.7445 |
| Support Vector Machine | 0.9650 | 0.9982 | 0.9650 | 0.9650 | 0.9650 | 0.9533 |

Best hyper-parameters found by 5-fold CV on the training split: Logistic Regression `C=100`;
Decision Tree `max_depth=12, min_samples_leaf=5`; KNN `k=21, weights=distance`; Naive Bayes
`var_smoothing=1e-9`; SVM `kernel=linear, C=10`.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best overall with SVM: 96.5 % accuracy, AUC 0.998, MCC 0.953. CV picked the weakest regularisation (C=100), so the price bands are close to linearly separable in the scaled feature space (RAM carries most of the signal). |
| Decision Tree | 85.8 % accuracy and MCC 0.81. It is good but clearly behind the linear models, because axis-aligned splits approximate the smooth, linear price boundary with many steps. Limiting depth and leaf size kept overfitting in check (CV 0.843 vs test 0.858). |
| KNN | Weakest model: 60.8 % accuracy, AUC 0.83, MCC 0.48. Only a few of the 20 features (RAM, battery, pixel size) really matter. Distance in the other 17 or so dimensions is mostly noise, so neighbours are not informative even after scaling (the curse of dimensionality). |
| Naive Bayes | 80.8 % accuracy, MCC 0.74. Decent for such a simple model, but the independence assumption is violated (px_height and px_width, or sc_h and sc_w, are related) and the binary features are not Gaussian. |
| Support Vector Machine | Ties with Logistic Regression (96.5 % accuracy, AUC 0.998). CV chose the linear kernel over RBF, which confirms the near-linear separability. It is slower and needs probability calibration for AUC. |
| **Overall winner for this dataset** | **Logistic Regression**: it ties SVM on every metric at 4 d.p. (SVM's AUC is 0.99815, which rounds to the same 0.9982) while being simpler, faster, and directly interpretable. SVM (linear) is an equally good alternative. |

## How to run

```bash
pip install -r requirements.txt
python train_models.py      # trains 5 models, writes model/ and test_data.csv
streamlit run app.py        # opens the app
```

The app lets you upload a test CSV (it must contain the 20 feature columns plus
`price_range`), choose a model from a dropdown, and view the six metrics, the confusion
matrix, the classification report, and a side-by-side table of all five models.

## Streamlit Community Cloud (if the app stays on “Your app is in the oven”)

Streamlit Cloud **does not read `runtime.txt`**. The build log shows the Python version
actually used (for example `Using CPython 3.14.10`). On Python 3.14 there is often **no
pre-built wheel for `scikit-learn==1.6.1`**, so the log shows `Building scikit-learn==1.6.1`
and the deploy can hang or fail for a long time.

**Fix:**

1. Delete the current Cloud app (or create a fresh deploy).
2. Deploy again from this repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** → **Python version** → choose **3.11** or **3.12** (not 3.14).
4. Save and deploy. The log should show `Using CPython 3.11.x` or `3.12.x` and **download**
   wheels for `scikit-learn`, not `Building scikit-learn`.

You cannot change Python on an existing app without deleting and redeploying. The saved
`.joblib` models in `model/` are already in the repo; the Cloud app only loads them (no
retraining on deploy).
