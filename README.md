# WANFIS Hydraulic System

Notebook utama: `notebook.ipynb`

Notebook ini membangun model **Wavelet-ANFIS (WANFIS)** untuk klasifikasi kondisi kebocoran pompa (`pump_leak`) pada dataset **Condition Monitoring of Hydraulic Systems**. Model menggunakan fitur hasil ekstraksi wavelet dari data sensor, lalu membentuk aturan fuzzy awal menggunakan **Subtractive Clustering** sebelum dilatih sebagai model ANFIS berbasis PyTorch.

## Tujuan

Tujuan utama sistem ini adalah:

- Mengolah data sensor hydraulic system menjadi fitur numerik yang ringkas.
- Mengklasifikasikan kondisi `pump_leak` ke dalam 3 kelas.
- Membangun model ANFIS dengan inisialisasi rule dari Subtractive Clustering.
- Melakukan tuning hyperparameter cluster radius (`r_a`) menggunakan Grid Search dan Stratified 5-Fold Cross Validation.
- Mengevaluasi performa model dengan accuracy, weighted F1-score, confusion matrix, dan akurasi per kelas.
- Mengekspor model terbaik agar bisa digunakan kembali.

## Dataset

Dataset didapatkan dari https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems.

Data sensor yang digunakan:

- `PS1`
- `PS2`
- `PS3`
- `TS1`
- `TS2`

Target klasifikasi diambil dari file `profile.txt`, yaitu kolom:

```text
pump_leak
```

Makna kelas:

| Kelas | Arti |
|---:|---|
| 0 | Tidak ada kebocoran atau normal |
| 1 | Kebocoran lemah |
| 2 | Kebocoran parah |

## Pipeline Notebook

Alur kerja notebook adalah sebagai berikut:

```text
Load data sensor
-> Ekstraksi fitur wavelet
-> Label encoding target pump_leak
-> Standardisasi fitur
-> Train-test split stratified
-> Subtractive Clustering
-> Inisialisasi ANFIS
-> Grid Search r_a + Stratified 5-Fold
-> Final training dengan best_r_a
-> Evaluasi model
-> Visualisasi hasil
-> Export model
```

## Ekstraksi Fitur Wavelet

Setiap sensor diproses menggunakan Discrete Wavelet Transform dengan wavelet `db4`.

Dari setiap sensor diambil 3 fitur:

- `mean_cA`: rata-rata koefisien approximation.
- `std_cA`: standar deviasi koefisien approximation.
- `energy_cD`: energi koefisien detail.

Karena ada 5 sensor dan setiap sensor menghasilkan 3 fitur, total fitur input adalah:

```text
5 sensor x 3 fitur = 15 fitur
```

Fitur ini menjadi input untuk model WANFIS.

## Subtractive Clustering

Subtractive Clustering digunakan untuk mencari pusat cluster pada ruang fitur yang sudah distandardisasi.

Setiap cluster center dipakai sebagai satu aturan fuzzy awal pada ANFIS:

```text
1 cluster center = 1 fuzzy rule
```

Hyperparameter penting pada tahap ini adalah `r_a` atau cluster radius.

Interpretasinya:

```text
r_a kecil  -> cluster lebih banyak -> rule lebih banyak -> model lebih kompleks
r_a besar  -> cluster lebih sedikit -> rule lebih sedikit -> model lebih sederhana
```

Karena jumlah rule berpengaruh langsung pada kompleksitas ANFIS, nilai `r_a` perlu dituning.

## Model ANFIS

Model ANFIS dibuat menggunakan PyTorch dengan struktur Sugeno sederhana:

- Layer 1: fuzzifikasi menggunakan Gaussian membership function.
- Layer 2: perhitungan firing strength rule.
- Layer 3: normalisasi firing strength.
- Layer 4: consequent Takagi-Sugeno berbasis kombinasi linear.
- Layer 5: output logits untuk klasifikasi multi-kelas.

Parameter membership function diinisialisasi dari hasil Subtractive Clustering:

- `mf_mean` dari cluster center.
- `mf_sigma` dari radius cluster.

Training menggunakan:

- Loss: `CrossEntropyLoss` dengan class weight.
- Optimizer: Adam.
- Learning rate membership function: `5e-3`.
- Learning rate consequent: `1e-2`.
- Scheduler: `CosineAnnealingLR`.
- Epoch: 200.

## Tuning Hyperparameter `r_a`

Notebook menambahkan Grid Search untuk mencoba beberapa kandidat `r_a`:

```python
ra_values = [0.3, 0.4, 0.5, 0.6, 0.7]
```

Setiap nilai `r_a` dievaluasi menggunakan **Stratified 5-Fold Cross Validation**. Stratified K-Fold digunakan agar proporsi setiap kelas `pump_leak` tetap seimbang pada setiap fold.

Untuk setiap fold:

1. Data train dan validasi dipisahkan.
2. Scaling dilakukan per fold untuk menghindari data leakage.
3. Subtractive Clustering dijalankan pada data train fold.
4. ANFIS dibuat berdasarkan jumlah rule fold tersebut.
5. Model dilatih dan dievaluasi pada validation fold.
6. Accuracy, weighted F1-score, dan jumlah rule dicatat.

Hasil tuning disimpan sebagai tabel:

```text
gridsearch_stratified_5fold_results.csv
```

Berdasarkan konteks eksperimen, nilai terbaik adalah:

```text
best_r_a = 0.6
```

Nilai ini dipilih karena menghasilkan rata-rata accuracy dan weighted F1-score tertinggi, dengan jumlah rule yang tetap efisien.

## Evaluasi dan Analisis

Notebook melakukan beberapa analisis evaluasi:

- Accuracy keseluruhan.
- Weighted F1-score.
- Classification report.
- Confusion matrix.
- Akurasi per kelas.
- Contoh probabilitas softmax untuk beberapa sampel test.
- Kurva training loss dan akurasi.
- Visualisasi membership function sebelum dan sesudah training.

Visualisasi yang dihasilkan:

```text
assets/images/training_curves.png
assets/images/evaluation_results.png
assets/images/membership_functions.png
assets/images/ra_experiment.png
```

Analisis utama yang dilihat:

- Apakah model mampu membedakan kelas normal, kebocoran lemah, dan kebocoran parah.
- Apakah jumlah fuzzy rule masih efisien.
- Apakah training stabil dari kurva loss dan akurasi.
- Bagaimana membership function berubah setelah optimasi gradient-based.

## Export Model

Model terbaik diekspor ke:

```text
models/anfis_subtractive_best_model.pth
```

Checkpoint menyimpan:

- `model_state_dict`
- `best_ra`
- `n_inputs`
- `n_rules`
- `n_classes`
- `feature_columns`
- `class_names`
- `label_encoder_classes`
- `scaler`
- `subtractive_centers`
- `subtractive_sigmas`
- `best_val_accuracy`

Ini penting karena model tidak cukup disimpan sebagai bobot saja. Saat inference, data baru harus diproses dengan urutan fitur dan scaler yang sama seperti saat training.

## Cara Menjalankan

1. Buka `notebook.ipynb`.
2. Pastikan folder `data/` berada di folder yang sama dengan notebook.
3. Jalankan cell dari awal sampai akhir.
4. Jalankan cell Grid Search untuk mendapatkan `best_ra`.
5. Lanjutkan cell final training.
6. Jalankan cell evaluasi dan export model.

Output akhir yang perlu dilaporkan:

```text
best_r_a
mean_accuracy +/- std_accuracy
mean_f1 +/- std_f1
mean_rules +/- std_rules
accuracy test
weighted F1-score test
jumlah rule final
confusion matrix
```
