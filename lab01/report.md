\# Lab 1 — Baseline Models and System Cost



\## 1. Goal



The goal of this laboratory was to build baseline machine learning models and measure their system costs. The measured costs included training time, single-sample inference latency, model size, and memory usage. The results were then used to evaluate whether the models are suitable for Cloud, Edge, Mobile, and TinyML deployment.



\## 2. Method



The Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn was used.



The dataset was divided into training and test sets using a 70/30 split. The split was stratified and `random\_state=42` was used for reproducibility.



Two baseline classification models were trained:



\* Logistic Regression with `max\_iter=1000` and `random\_state=42`

\* Random Forest with `n\_estimators=100` and `random\_state=42`



Test accuracy was measured for both models.



For system cost measurements, one warm-up training run was performed first. Training time was then measured over 5 runs and the median was reported. Single-sample inference latency was measured over 100 repetitions and the median was reported.



The models were saved using joblib to measure their file size. Memory usage was measured using the process RSS value provided by `psutil`.



\## 3. Results



\### Accuracy



| Model               | Test Accuracy |

| ------------------- | ------------: |

| Logistic Regression |        0.9415 |

| Random Forest       |        0.9357 |



\### System cost



| Model               | Training Time Median (s) | Inference Latency Median (ms) | Model Size (KB) | Memory (KB) |

| ------------------- | -----------------------: | ----------------------------: | --------------: | ----------: |

| Logistic Regression |                 0.613653 |                      0.087100 |            1.03 |   106856.00 |

| Random Forest       |                 0.151398 |                      2.297150 |          284.07 |   108160.00 |



\### Deployment suitability



| Deployment | Requirements                                       | Logistic Regression | Random Forest |

| ---------- | -------------------------------------------------- | ------------------- | ------------- |

| Cloud      | Memory ≥ 1 GB, latency ≤ 100 ms, model ≤ 500 MB    | Suitable            | Suitable      |

| Edge       | Memory 256–1024 MB, latency ≤ 50 ms, model ≤ 50 MB | Suitable            | Suitable      |

| Mobile     | Memory 64–256 MB, latency ≤ 20 ms, model ≤ 10 MB   | Suitable            | Suitable      |

| TinyML     | Memory ≤ 256 KB, latency ≤ 10 ms, model ≤ 100 KB   | Not suitable        | Not suitable  |



The measured process memory is much larger than the model file size because RSS measures the memory used by the Python process, including the libraries and runtime. Therefore, the models themselves are much smaller than the reported process memory.



\## 4. Conclusions



1\. Logistic Regression achieved slightly higher accuracy than Random Forest, with 0.9415 compared with 0.9357.



2\. Random Forest had a shorter training time, but Logistic Regression had much lower inference latency and a much smaller model size. Therefore, Logistic Regression is more efficient when fast inference and low storage requirements are important.



3\. Both models are suitable for Cloud, Edge, and Mobile deployment according to the measured latency and model size. They are not suitable for TinyML because the measured process memory is much higher than the TinyML memory budget.



