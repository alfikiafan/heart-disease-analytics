# Prediksi Risiko Penyakit Jantung

Proyek ini bertujuan untuk menganalisis dan mengklasifikasikan risiko penyakit jantung menggunakan beberapa algoritma machine learning. Dengan menggunakan dataset yang terdiri dari 1190 sampel dan 12 fitur yang mencakup informasi demografis serta kesehatan pasien, proyek ini berfokus pada pengklasifikasian pasien sebagai Berisiko Tinggi atau Tidak Berisiko terhadap penyakit jantung. Model yang digunakan meliputi Logistic Regression dan Decision Tree, dengan evaluasi performa menggunakan berbagai metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC.

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Business Understanding](#business-understanding)
  - [Problem Statement](#problem-statement)
  - [Goals](#goals)
  - [Solution Statement](#solution-statement)
- [Data Understanding](#data-understanding)
  - [Deskripsi Dataset](#deskripsi-dataset)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preparation](#data-preparation)
  - [Encoding Kategorikal](#encoding-kategorikal)
  - [Feature Scaling](#feature-scaling)
  - [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree Classifier](#decision-tree-classifier)
- [Evaluasi](#evaluasi)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Cross-Validation](#cross-validation)
  - [Evaluasi Lanjutan](#evaluasi-lanjutan)
  - [Menyimpan dan Memuat Model Terbaik](#menyimpan-dan-memuat-model-terbaik)
- [Visualisasi Hasil](#visualisasi-hasil)
  - [Perbandingan Metrik Model](#perbandingan-metrik-model)
  - [ROC Curves untuk Semua Model](#roc-curves-untuk-semua-model)
  - [Confusion Matrix untuk Semua Model](#confusion-matrix-untuk-semua-model)
  - [Feature Importance (Decision Tree)](#feature-importance-decision-tree)
  - [Ringkasan Metrik Evaluasi dalam Tabel](#ringkasan-metrik-evaluasi-dalam-tabel)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Dependensi](#dependensi)

## Pendahuluan

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Dengan meningkatnya kasus penyakit jantung, penting untuk dapat mengidentifikasi faktor-faktor risiko dan mengklasifikasikan pasien berdasarkan risiko mereka. Proyek ini menggunakan teknik machine learning untuk membangun model prediksi yang dapat membantu tenaga medis dalam pengambilan keputusan.

## Business Understanding

### Problem Statement

Bagaimana memprediksi risiko penyakit jantung pada pasien berdasarkan data kesehatan mereka?

### Goals

- Membangun model klasifikasi yang akurat untuk mengidentifikasi pasien berisiko tinggi penyakit jantung.
- Memilih model yang mudah diinterpretasikan untuk mendukung pengambilan keputusan medis.

### Solution Statement

1. Menggunakan Logistic Regression sebagai model baseline.
2. Menggunakan Decision Tree Classifier untuk menangkap hubungan non-linear antar fitur.
3. Melakukan hyperparameter tuning untuk meningkatkan performa model.

## Alur Kode

### 1. Data Understanding

#### Exploratory Data Analysis (EDA)

Proses EDA meliputi:
- Memeriksa informasi dasar dataset.
- Menampilkan statistik deskriptif.
- Menganalisis distribusi fitur numerik dan kategorikal.
- Membuat matriks korelasi dan pairplot.
- Memeriksa missing values.

### 2. Data Preparation

#### Encoding Kategorikal

Fitur kategorikal seperti `chest pain type`, `resting ecg`, dan `ST slope` diubah menjadi format numerik menggunakan One-Hot Encoding.

#### Feature Scaling

Fitur numerik seperti `age`, `resting bp s`, dan `cholesterol` di-standardisasi menggunakan `StandardScaler` untuk memastikan setiap fitur memiliki rata-rata 0 dan deviasi standar 1.

#### Feature Engineering

Membuat fitur baru yang merupakan interaksi antara `age` dan `cholesterol` untuk menangkap hubungan kompleks antara kedua fitur tersebut.

### 3. Modeling

#### Logistic Regression

Model Logistic Regression digunakan sebagai baseline karena kemudahan interpretasinya dalam memahami hubungan antara fitur dan target.

#### Decision Tree Classifier

Model Decision Tree dipilih untuk interpretasi visual yang baik dan kemampuan menangkap hubungan non-linear antar fitur.

#### Hyperparameter Tuning

Melakukan hyperparameter tuning pada kedua model menggunakan `GridSearchCV` untuk menemukan kombinasi parameter terbaik yang meningkatkan performa model.

#### Cross-Validation

Melakukan cross-validation dengan 5 fold untuk memastikan model tidak overfitting dan memiliki performa yang konsisten.

### 4. Evaluasi

Menampilkan kembali classification report dan ROC-AUC score setelah cross-validation dan hyperparameter tuning.

#### Menyimpan dan Memuat Model Terbaik

Model terbaik berdasarkan ROC-AUC disimpan menggunakan `joblib` dan dimuat kembali untuk memastikan bahwa model bekerja dengan baik setelah disimpan.

#### Perbandingan Metrik Model

Membandingkan metrik evaluasi (Akurasi, Precision, Recall, F1-Score, ROC-AUC) antara Logistic Regression dan Decision Tree menggunakan bar plot.

#### ROC Curves untuk Semua Model

Menampilkan ROC Curve untuk kedua model dalam satu plot untuk memudahkan perbandingan kemampuan masing-masing model.

#### Confusion Matrix untuk Semua Model

Menampilkan Confusion Matrix untuk kedua model secara berdampingan untuk melihat distribusi prediksi mereka.

#### Feature Importance (Decision Tree)

Menampilkan fitur-fitur yang paling penting menurut model Decision Tree untuk memahami fitur mana yang paling berpengaruh dalam prediksi risiko penyakit jantung.

#### Ringkasan Metrik Evaluasi dalam Tabel

Membuat tabel ringkasan metrik evaluasi untuk kedua model agar memudahkan perbandingan.

## Instalasi

Pastikan Anda telah menginstal Python 3.7 atau yang lebih baru. Instalasi dapat dilakukan dengan mengikuti langkah-langkah berikut:

1. **Clone Repository:**
   ```bash
   git clone https://github.com/username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Buat Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # Untuk Windows: env\Scripts\activate
   ```

3. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

## Cara Penggunaan

1. **Persiapan Data:**  
   Pastikan file dataset `heart_disease_dataset.csv` berada di direktori proyek.

2. **Jalankan Script:**  
   Anda dapat menjalankan script analisis menggunakan Jupyter Notebook atau langsung melalui Python.  
   Jika menggunakan Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
     Buka notebook yang relevan dan jalankan sel-sel kode.

3. **Evaluasi Model:**  
   Setelah menjalankan seluruh script, model terbaik akan disimpan sebagai `model_terbaik.pkl`.  
   Anda dapat memuat dan menggunakan model tersebut untuk prediksi lebih lanjut.

## Dependensi

Proyek ini memerlukan beberapa library Python berikut:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib

*Terima kasih telah menggunakan proyek Prediksi Risiko Penyakit Jantung ini! Semoga bermanfaat dalam upaya pencegahan dan penanganan penyakit jantung.*