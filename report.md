# Laporan Proyek Machine Learning

## Domain Proyek

Proyek ini mengambil domain kesehatan dengan judul "Klasifikasi Risiko Penyakit Jantung dengan Machine Learning".

### Latar Belakang

Penyakit jantung adalah salah satu masalah kesehatan global yang signifikan, menyebabkan jutaan kematian setiap tahunnya. Menurut World Health Organization (WHO), penyakit kardiovaskular, yang mencakup penyakit seperti penyakit jantung koroner, serangan jantung, dan stroke merupakan penyebab utama kematian di dunia, dengan estimasi jumlah kematian sebesar 17,9 juta orang di seluruh dunia pada tahun 2019 [[1](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))]. Di Indonesia, prevalensi penyakit jantung juga menunjukkan peningkatan yang signifikan seiring dengan perubahan gaya hidup dan peningkatan faktor risiko seperti hipertensi, diabetes, dan obesitas [[2](https://www.kemkes.go.id/id/profil-kesehatan-indonesia-2023)].

Deteksi dini risiko penyakit jantung sangat penting untuk mencegah perkembangan penyakit lebih lanjut dan meningkatkan kualitas hidup pasien. Metode tradisional dalam diagnosis penyakit jantung seringkali memerlukan waktu dan sumber daya yang cukup besar, serta bergantung pada interpretasi subjektif dari tenaga medis. Oleh karena itu, penggunaan analitik prediktif berbasis machine learning menawarkan solusi yang lebih efisien dan objektif dalam mengidentifikasi pasien berisiko tinggi [[3](https://repository.bsi.ac.id/index.php/unduh/item/344564/01-Nugraha---Prediksi-Penyakit-Jantung-Cardiovascular-Menggunakan-Model-Algoritma-Klasifikasi.pdf)].

Beberapa penelitian telah menunjukkan keberhasilan model machine learning dalam memprediksi risiko penyakit jantung. Misalnya, penelitian oleh Pratama Hariyono (2021) menunjukkan bahwa algoritma **Naïve Bayes** dan **Decision Tree** dapat memberikan akurasi yang tinggi dalam klasifikasi risiko penyakit jantung [[4](https://repository.unair.ac.id/102857/)]. Selain itu, penelitian oleh Bukhari et al. (2023) membandingkan metode **Decision Tree** dan **Regresi Logistik** dalam mendeteksi penyakit jantung, dengan hasil bahwa regresi logistik memiliki akurasi yang lebih tinggi [[5](https://e-journals.unmul.ac.id/index.php/jsakti/article/viewFile/10780/pdf)].

Studi lain oleh Alhamad et al. (2019) menggarisbawahi pentingnya pemilihan fitur yang tepat dan teknik **feature engineering** dalam meningkatkan performa model prediksi [[6](https://jurnal.untan.ac.id/index.php/jepin/article/view/37188)]. Penggunaan library **Scikit-learn** telah mempermudah implementasi berbagai algoritma machine learning dan teknik evaluasi, memungkinkan peneliti untuk dengan cepat mengembangkan dan menguji model prediktif [[7](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)].

Meskipun metode lain seperti **Convolutional Neural Networks (CNNs)** telah diterapkan dalam berbagai domain, termasuk klasifikasi teks, model-model tersebut sering kali memerlukan sumber daya komputasi yang lebih besar dan interpretasi yang lebih kompleks [[8](https://aclanthology.org/I17-1026.pdf)]. Oleh karena itu, dalam konteks proyek ini, pemilihan **Logistic Regression** dan **Decision Tree** dianggap lebih sesuai karena keseimbangan antara akurasi, interpretabilitas, dan efisiensi.

Untuk mengatasi masalah ini, proyek ini bertujuan untuk membangun model klasifikasi yang akurat dan mudah diinterpretasikan untuk mengidentifikasi pasien berisiko tinggi penyakit jantung. Dengan menggunakan dataset yang terdiri dari berbagai fitur kesehatan dan demografis, proyek ini akan menerapkan algoritma **Logistic Regression** sebagai model baseline dan **Decision Tree** untuk interpretasi visual. Selain itu, proses **hyperparameter tuning** akan dilakukan untuk meningkatkan performa model. Meskipun demikian, analisis lanjutan oleh dokter tetap perlu dilakukan untuk mengonfirmasi diagnosis awal dan rekomendasi pengobatan yang sesuai.

## Business Understanding

Dalam proyek ini, penulis mengklasifikasikan risiko penyakit jantung menjadi dua kategori: **berisiko (1)** dan **tidak berisiko (0)**.

### Problem Statements

Untuk membangun model klasifikasi risiko penyakit jantung yang efektif, terdapat beberapa pertanyaan kunci yang perlu dijawab:

1. Bagaimana cara mengklasifikasikan pasien sebagai berisiko atau tidak berisiko penyakit jantung berdasarkan data kesehatan dan demografis yang tersedia?
   
2. Algoritma machine learning mana yang paling efektif dalam mengklasifikasikan pasien sebagai berisiko tinggi atau tidak berisiko penyakit jantung berdasarkan dataset yang tersedia?
   
3. Bagaimana meningkatkan akurasi model klasifikasi melalui optimasi parameter dan teknik pemodelan lanjutan?
   
4. Sejauh mana model yang dibangun dapat diinterpretasikan untuk mendukung pengambilan keputusan medis yang lebih baik?

### Goals

Berdasarkan perumusan masalah di atas, tujuan proyek ini adalah sebagai berikut:

1. **Membangun model klasifikasi yang akurat** untuk mengidentifikasi pasien berisiko tinggi penyakit jantung berdasarkan data kesehatan dan demografis.
   
2. **Memilih dan mengimplementasikan beberapa algoritma machine learning** yang efektif, seperti Logistic Regression dan Decision Tree, untuk mencapai klasifikasi yang optimal.
   
3. **Melakukan optimasi parameter (hyperparameter tuning)** pada model yang dipilih untuk meningkatkan performa klasifikasi.
   
4. **Mengevaluasi dan membandingkan performa berbagai model** menggunakan metrik evaluasi yang sesuai, seperti akurasi, precision, recall, F1-score, dan ROC-AUC, guna menentukan model terbaik yang dapat diandalkan dalam konteks medis.

### Solution Statement

Untuk mencapai tujuan-tujuan tersebut, proyek ini akan menerapkan beberapa solusi strategis sebagai berikut:

1. **Menggunakan Logistic Regression dan Decision Tree untuk Klasifikasi Risiko Penyakit Jantung:**
   - Kedua algoritma ini dipilih karena kemampuan mereka dalam menangani masalah klasifikasi biner serta interpretabilitas yang baik.
   - **Manfaat:** Logistic Regression menyediakan interpretasi koefisien yang jelas terkait kontribusi setiap fitur terhadap probabilitas risiko, sementara Decision Tree menawarkan struktur keputusan visual yang memudahkan pemahaman proses klasifikasi.
   
2. **Melakukan Hyperparameter Tuning untuk Meningkatkan Performa Model:**
   - Optimasi parameter model melalui teknik seperti Grid Search akan diterapkan pada kedua algoritma untuk menemukan kombinasi parameter terbaik yang meningkatkan akurasi dan metrik evaluasi lainnya.
   - **Manfaat:** Meningkatkan performa model secara signifikan tanpa harus mengganti algoritma yang digunakan, memastikan bahwa solusi yang diusulkan mencapai tingkat akurasi yang optimal.
   
3. **Menerapkan Cross-Validation dan Teknik Evaluasi Lanjutan:**
   - Menggunakan metode cross-validation untuk memastikan bahwa model tidak mengalami overfitting dan memiliki performa yang konsisten pada data yang berbeda.
   - **Manfaat:** Menjamin keandalan model dalam berbagai kondisi data, sehingga hasil prediksi dapat dipercaya untuk diterapkan dalam konteks medis.
   
4. **Membandingkan Hasil dari Berbagai Model setelah Hyperparameter Tuning:**
   - Setelah melakukan hyperparameter tuning, performa Logistic Regression dan Decision Tree akan dibandingkan menggunakan metrik evaluasi yang telah ditentukan, yakni akurasi, precision, recall, F1-score, dan ROC-AUC.
   - **Manfaat:** Memilih model terbaik berdasarkan performa yang diukur, sehingga memastikan bahwa model yang dipilih adalah yang paling efektif dalam mengklasifikasikan risiko penyakit jantung.

## Data Understanding
### Informasi Dataset

Link Dataset: https://www.kaggle.com/datasets/mexwell/heart-disease-dataset

Proyek ini menggunakan dataset **🫀 Heart Disease Dataset** yang dapat diakses melalui [Kaggle](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset). Dataset ini dikurasi dengan menggabungkan lima dataset penyakit jantung populer yang sebelumnya tersedia secara independen namun belum pernah digabungkan sebelumnya. Kelima dataset tersebut adalah:

1. Cleveland
2. Hungarian
3. Switzerland
4. Long Beach VA
5. Statlog (Heart) Data Set

Dengan penggabungan ini, dataset menjadi yang terbesar dengan total **1190** sampel dan **12** fitur yang mencakup informasi demografis dan kesehatan pasien.

### Deskripsi Atribut Dataset

Berikut adalah deskripsi lengkap dari setiap atribut dalam dataset:

| No. | Attribute Code | Deskripsi                                                                                                 | Tipe Data | Unit       |
|-------|----------------|-----------------------------------------------------------------------------------------------------------|-----------|------------|
| 1     | age            | Umur pasien dalam tahun                                                                                   | Numeric   | Tahun      |
| 2     | sex            | Jenis kelamin pasien (1 = pria, 0 = wanita)                                                              | Binary    | -          |
| 3     | chest pain type | Tipe nyeri dada (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)          | Nominal   | -          |
| 4     | resting bp s   | Tekanan darah istirahat dalam mm Hg                                                                        | Numeric   | mm Hg      |
| 5     | cholesterol    | Kadar kolesterol serum dalam mg/dl                                                                           | Numeric   | mg/dl      |
| 6     | fasting blood sugar | Apakah gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak)                                            | Binary    | mg/dl      |
| 7     | resting ecg    | Hasil elektrokardiogram istirahat (0: normal, 1: ST-T wave abnormality, 2: probable atau definite LVH) | Nominal   | -          |
| 8     | max heart rate | Denyut jantung maksimum yang dicapai                                                                         | Numeric   | bpm        |
| 9     | exercise angina | Angina yang diinduksi oleh latihan (1 = ya, 0 = tidak)                                                  | Binary    | -          |
| 10    | oldpeak        | Depresi ST setelah latihan                                                                                 | Numeric   | -          |
| 11    | ST slope       | Kemiringan segmen ST pada latihan (1: upsloping, 2: flat, 3: downsloping)                                | Nominal   | -          |
| 12    | class          | Status penyakit jantung (1 = penyakit jantung, 0 = normal)                                               | Binary    | -          |

### Exploratory Data Analysis - EDA

Pada tahap ini, dilakukan analisis eksploratif untuk memahami struktur dan karakteristik dasar data sebagai berikut.

#### 1. Pemeriksaan Dimensi dan Tipe Data:
- Menggunakan `data.info()` untuk melihat jumlah baris dan kolom, serta tipe data dari setiap variabel. 
    ```text
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1190 entries, 0 to 1189
    Data columns (total 12 columns):
    #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
    0   age                  1190 non-null   int64  
    1   sex                  1190 non-null   int64  
    2   chest pain type      1190 non-null   int64  
    3   resting bp s         1190 non-null   int64  
    4   cholesterol          1190 non-null   int64  
    5   fasting blood sugar  1190 non-null   int64  
    6   resting ecg          1190 non-null   int64  
    7   max heart rate       1190 non-null   int64  
    8   exercise angina      1190 non-null   int64  
    9   oldpeak              1190 non-null   float64
    10  ST slope             1190 non-null   int64  
    11  target               1190 non-null   int64  
    dtypes: float64(1), int64(11)
    memory usage: 111.7 KB
    ```
    Fungsi ini digunakan untuk mengetahui jumlah baris dan kolom dalam dataset, serta tipe data dari setiap variabel. Dari output, terlihat bahwa dataset terdiri dari 1190 baris dan 12 kolom dengan tipe data numerik.
- Menggunakan `data.head()` untuk melihat 5 baris pertama dari dataset. Kegunaannya adalah untuk:  

  - Memahami struktur data dan tipe variabel.
  - Memastikan bahwa kolom dan data yang relevan.
  - Mengetahui format data dan tipe variabel.  

  ![Informasi Dataset](/img/2.1.png)

#### 2. Distribusi Fitur Numerik

Visualisasi distribusi fitur numerik dilakukan menggunakan histogram `sns.histplot()` dan boxplot `sns.boxplot()` untuk memahami sebaran data dan mengidentifikasi outliers.

![Distribusi Fitur age](/img/2.2.png)
![Distribusi Fitur resting bp s](/img/2.3.png)
![Distribusi Fitur cholesterol](/img/2.4.png)
![Distribusi Fitur max heart rate](/img/2.5.png)
![Distribusi Fitur oldpeak](/img/2.6.png)

**Temuan:**
Berikut adalah insight untuk masing-masing fitur berdasarkan visualisasi histogram dan boxplot:

**a. Age:**
   - Distribusi usia terlihat simetris dengan puncak di sekitar usia 50-60 tahun, menunjukkan bahwa sebagian besar pasien dalam dataset berada dalam rentang usia tersebut.
   - Tidak ada outlier yang signifikan pada boxplot, menunjukkan distribusi usia yang cukup normal.

**b. Resting Blood Pressure (resting bp s):**
   - Distribusi tekanan darah saat istirahat memiliki puncak sekitar 120-140 mmHg, dengan skew ke arah nilai yang lebih tinggi.
   - Terdapat beberapa outlier yang signifikan di sisi kanan boxplot (tekanan darah tinggi), serta satu nilai yang sangat rendah. Ini mungkin menunjukkan pasien dengan kondisi ekstrem, seperti hipertensi berat atau tekanan darah sangat rendah.

**c. Cholesterol:**
   - Distribusi kolesterol menunjukkan skew ke kanan dengan puncak di sekitar 200-250 mg/dL.
   - Boxplot menunjukkan beberapa outlier, terutama pada nilai kolesterol yang sangat tinggi (>400 mg/dL), yang mungkin menunjukkan risiko penyakit jantung yang lebih tinggi pada pasien dengan kadar kolesterol tersebut.

**d. Max Heart Rate Achieved:**
   - Distribusi detak jantung maksimum cukup simetris, dengan puncak di sekitar 130-150 bpm.
   - Terdapat sedikit outlier pada nilai rendah di sekitar 60 bpm, mungkin menunjukkan pasien dengan kondisi jantung tertentu atau keterbatasan dalam mencapai detak jantung maksimal.

**e. Oldpeak:**
   - Distribusi oldpeak memiliki skew ke kanan, dengan sebagian besar nilai mendekati 0. Oldpeak ini menunjukkan depresi ST setelah olahraga, yang menjadi indikator kesehatan jantung.
   - Beberapa outlier terlihat pada nilai yang lebih tinggi (3-6), yang mungkin menunjukkan pasien dengan gangguan jantung yang signifikan.

#### 3. Distribusi Fitur Kategorikal

Visualisasi distribusi fitur kategorikal menggunakan `sns.countplot()` untuk memahami proporsi setiap kategori.

![Distribusi jenis kelamin](/img/2.7.png)
![Distribusi tipe nyeri dada](/img/2.8.png)
![Distribusi gula darah puasa](/img/2.9.png)
![Distribusi hasil EKG istirahat](/img/2.10.png)
![Distribusi angina yang muncul saat olahraga](/img/2.11.png)
![Distribusi kemiringan segmen ST](/img/2.12.png)
![Distribusi status penyakit jantung](/img/2.13.png)

**Temuan:**
Berikut adalah insight untuk masing-masing fitur kategorikal berdasarkan visualisasi countplot:

1. **Sex (Jenis Kelamin):**
   - Mayoritas data pasien adalah laki-laki (kode 1), dengan jumlah yang jauh lebih tinggi dibandingkan perempuan (kode 0).
   - Perbedaan ini bisa relevan untuk analisis lebih lanjut, karena penyakit jantung sering kali menunjukkan prevalensi yang berbeda antara pria dan wanita.

2. **Chest Pain Type:**
   - Sebagian besar pasien mengalami tipe nyeri dada 4 (asymptomatic), yang menunjukkan kondisi tanpa gejala.
   - Tipe nyeri dada ini dapat memiliki korelasi yang signifikan dengan risiko penyakit jantung, sehingga penting untuk diperhatikan dalam pemodelan.

3. **Fasting Blood Sugar:**
   - Sebagian besar pasien memiliki kadar gula darah puasa di bawah 120 mg/dL (kode 0), sementara sebagian kecil memiliki kadar gula tinggi (kode 1).
   - Tingginya gula darah puasa sering dikaitkan dengan risiko penyakit kardiovaskular, sehingga fitur ini bisa menjadi indikator penting.

4. **Resting ECG:**
   - Sebagian besar pasien menunjukkan hasil EKG normal (kode 0), diikuti oleh hasil yang menunjukkan kemungkinan hipertrofi ventrikel kiri (kode 2).
   - Hasil EKG yang tidak normal dapat menunjukkan masalah pada jantung, sehingga penting dalam mendeteksi risiko penyakit jantung.

5. **Exercise Angina:**
   - Sebagian besar pasien tidak mengalami angina saat berolahraga (kode 0), sementara sekitar sepertiga pasien mengalaminya (kode 1).
   - Angina yang terjadi saat berolahraga sering menjadi indikator kondisi jantung yang buruk, sehingga penting dalam analisis risiko.

6. **ST Slope:**
   - Mayoritas pasien memiliki ST slope jenis 1 (upsloping) dan 2 (flat), dengan sedikit pasien yang memiliki jenis slope 3 (downsloping).
   - ST slope adalah indikator penting pada hasil EKG yang dapat menunjukkan risiko penyakit jantung.

7. **Target:**
   - Target data cukup seimbang antara pasien yang berisiko penyakit jantung (kode 1) dan yang tidak berisiko (kode 0).
   - Keseimbangan ini ideal untuk pemodelan karena mengurangi potensi bias dalam prediksi.

#### 4. Matriks Korelasi

Membuat heatmap dari matriks korelasi untuk melihat hubungan antar fitur numerik dengan menggunakan `sns.heatmap()`.

![Matriks Korelasi](/img/2.14.png)

**Temuan:**
Berikut adalah beberapa insight dari **Correlation Matrix** terkait risiko penyakit jantung:

**a. Fitur dengan Korelasi Tinggi terhadap Target:**
   - **Chest Pain Type (0.46)**: Tipe nyeri dada memiliki korelasi positif yang relatif tinggi dengan risiko penyakit jantung. Artinya, jenis nyeri dada tertentu lebih sering muncul pada pasien yang berisiko.
   - **Exercise Angina (0.48)**: Angina yang dialami saat berolahraga memiliki korelasi yang cukup tinggi dengan target, menunjukkan bahwa pasien dengan angina saat berolahraga cenderung memiliki risiko penyakit jantung.
   - **Oldpeak (0.40)** dan **ST Slope (0.51)**: Kedua fitur ini menunjukkan korelasi yang signifikan dengan target. Oldpeak, yang menunjukkan depresi ST, serta tipe ST slope, dapat menjadi indikator penting dalam mendeteksi risiko penyakit jantung.

**b. Fitur dengan Korelasi Negatif terhadap Target:**
   - **Max Heart Rate (-0.41)**: Detak jantung maksimum memiliki korelasi negatif dengan risiko penyakit jantung, menunjukkan bahwa pasien dengan detak jantung maksimum yang lebih tinggi cenderung memiliki risiko lebih rendah.
   - **Age (0.26)** dan **Sex (0.31)**: Korelasi ini menunjukkan bahwa risiko penyakit jantung sedikit lebih tinggi pada laki-laki dan pasien yang lebih tua, namun pengaruhnya tidak sekuat fitur lainnya.

**c. Korelasi Antar Fitur:**
   - **Oldpeak dan ST Slope (0.52)**: Korelasi positif yang kuat antara Oldpeak dan ST Slope menunjukkan bahwa pasien dengan nilai Oldpeak tinggi cenderung memiliki tipe ST slope tertentu. Ini bisa menjadi indikator komorbiditas atau karakteristik kondisi jantung.
   - **Max Heart Rate dan Age (-0.37)**: Ada korelasi negatif antara detak jantung maksimum dan usia, yang sesuai dengan fisiologi umum di mana detak jantung maksimum menurun seiring bertambahnya usia.

**d. Insight Utama untuk Pemodelan:**
   - Fitur seperti **Chest Pain Type**, **Exercise Angina**, **Oldpeak**, dan **ST Slope** menunjukkan korelasi tinggi dengan target dan kemungkinan besar akan berkontribusi besar dalam proses pemodelan.
   - Meskipun fitur seperti **Age** dan **Sex** menunjukkan korelasi dengan target, kontribusinya relatif lebih rendah dibandingkan fitur lain yang lebih berkaitan langsung dengan kondisi jantung.

#### 5. Pairplot

Membuat pairplot untuk memvisualisasikan hubungan antar fitur dan target dengan menggunakan `sns.pairplot()`.

![Pairplot](/img/2.15.png)

**Temuan:**
Berikut adalah beberapa insight dari **Pairplot** yang menunjukkan hubungan antar fitur berdasarkan target (0: tidak memiliki penyakit jantung, 1: memiliki penyakit jantung):

**a. Distribusi Target Berdasarkan Fitur:**
   - **Chest Pain Type:** Pasien dengan penyakit jantung (target = 1) memiliki kecenderungan lebih tinggi mengalami tipe nyeri dada tertentu, terutama pada tipe 4 (asymptomatic).
   - **Max Heart Rate:** Pasien normal cenderung memiliki detak jantung maksimum yang lebih tinggi dibandingkan pasien berisiko. Hal ini terlihat dari distribusi titik yang lebih terkonsentrasi pada nilai max heart rate yang lebih rendah untuk target = 1.
   - **Oldpeak:** Pasien dengan penyakit jantung memiliki nilai oldpeak yang lebih tinggi, yang menunjukkan depresi ST yang lebih signifikan saat berolahraga. Distribusi untuk target = 1 menunjukkan konsentrasi pada nilai oldpeak yang lebih besar.

**b. Kecenderungan Hubungan Antar Fitur:**
   - **Oldpeak dan ST Slope:** Terdapat hubungan yang kuat antara oldpeak dan ST slope, yang ditunjukkan dengan distribusi yang jelas pada pasien berisiko (target = 1). Pasien dengan nilai oldpeak tinggi cenderung memiliki tipe ST slope tertentu (flat atau downsloping), yang bisa menjadi penanda kondisi jantung yang lebih serius.
   - **Max Heart Rate dan Age:** Detak jantung maksimum cenderung menurun seiring bertambahnya usia, terlihat dari distribusi titik yang menurun di kolom Max Heart Rate seiring dengan bertambahnya nilai Age.

**c. Pemisahan Berdasarkan Target:**
   - **Fitur-fitur seperti Chest Pain Type, Exercise Angina, dan ST Slope** menunjukkan pemisahan yang cukup baik antara pasien yang berisiko (target = 1) dan tidak berisiko (target = 0), yang mungkin menandakan bahwa fitur-fitur ini bisa menjadi penanda penting untuk klasifikasi.
   - **Kolom Fasting Blood Sugar dan Resting ECG:** Kedua fitur ini tidak menunjukkan pemisahan yang jelas antara target, yang mengindikasikan bahwa mungkin fitur ini kurang berkontribusi dalam pemodelan dibandingkan fitur lain yang menunjukkan pemisahan lebih baik.

**d. Konsentrasi Nilai pada Beberapa Fitur:**
   - Beberapa fitur, seperti **Resting BP S** dan **Cholesterol**, memiliki distribusi yang hampir sama pada kedua kelas target, yang menunjukkan kurangnya kemampuan fitur ini untuk membedakan pasien berisiko dari yang tidak berisiko.

#### 6. Pemeriksaan Missing Values

Untuk memeriksa apakah terdapat nilai yang hilang dalam dataset, digunakan `data.isnull().sum()`.

```text
                    0
age                 0
sex                 0
chest pain type     0
resting bp s        0
cholesterol         0
fasting blood sugar 0
resting ecg         0
max heart rate      0
exercise angina     0
oldpeak             0
ST slope            0
target              0

dtype: int64
```

**Hasil:**
- Tidak terdapat nilai yang hilang dalam dataset.  

### 7. Outlier
Outlier adalah nilai yang jauh berbeda dari mayoritas data dan dapat mempengaruhi hasil analisis serta performa model machine learning. Untuk memastikan kualitas data yang optimal, penghapusan outlier biasanya dipertimbangkan. Namun, dalam analisis ini, penghapusan outlier menggunakan metode Z-Score dievaluasi tetapi diputuskan untuk tidak dilakukan karena tidak meningkatkan performa model.

**Rumus Z-Score**

Z-Score digunakan untuk mengukur seberapa jauh sebuah nilai dari rata-rata dalam satuan deviasi standar. Rumus Z-Score adalah sebagai berikut:

$$z = \frac{(x - \mu)}{\sigma}$$

di mana:
- $x$ = nilai individu
- $\mu$ = rata-rata dataset
- $\sigma$ = deviasi standar dataset

**Alasan Tidak Menghapus Outlier**

- Setelah melakukan analisis, ditemukan bahwa penghapusan outlier tidak memberikan peningkatan performa model. Sebaliknya, model menunjukkan hasil yang lebih baik tanpa menghapus outlier, mungkin karena outlier tersebut merepresentasikan kasus-kasus penting yang berkontribusi pada pemahaman risiko penyakit jantung.  

- Menghapus outlier dapat menyebabkan hilangnya informasi penting yang sebenarnya relevan dalam konteks medis, seperti pasien dengan kondisi ekstrem yang memerlukan perhatian khusus. 

## Data Preparation

Pada tahap ini, data yang telah dipahami sebelumnya akan dipersiapkan agar siap digunakan dalam proses pemodelan machine learning. Proses persiapan data meliputi encoding fitur kategorikal, feature scaling, dan feature engineering. Setiap langkah dilakukan dengan alasan yang jelas untuk memastikan kualitas data yang optimal dan meningkatkan performa model yang akan dibangun. 

### 1. Encoding Fitur Kategorikal

Fitur kategorikal seperti `chest pain type`, `resting ecg`, dan `ST slope` perlu diubah menjadi format numerik agar dapat diproses oleh algoritma machine learning. Salah satu metode yang umum digunakan adalah **One-Hot Encoding**.

**Alasan melakukan encoding**

- Banyak algoritma machine learning hanya dapat bekerja dengan data numerik.
- One-Hot Encoding mencegah model menganggap adanya hubungan ordinal antara kategori yang sebenarnya tidak ada.

**Implementasi Kode**

```python
categorical_features = ['chest pain type', 'resting ecg', 'ST slope']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

print(f"Jumlah fitur setelah One-Hot Encoding: {data.shape[1]}")
```

### 2. Feature Scaling

Feature scaling adalah proses menyesuaikan skala fitur numerik sehingga setiap fitur memiliki kontribusi yang seimbang dalam proses pemodelan. **StandardScaler** adalah salah satu metode yang umum digunakan untuk melakukan feature scaling dengan mengubah data sehingga memiliki mean 0 dan standard deviation 1.

#### Alasan melakukan feature scaling

- Feature scaling memastikan bahwa fitur dengan skala yang lebih besar tidak mendominasi proses pembelajaran model.
- Banyak algoritma machine learning, seperti Logistic Regression dan Neural Networks, bekerja lebih baik dengan data yang telah di-scale.

#### Implementasi Kode

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
print("Feature scaling selesai dilakukan menggunakan StandardScaler.")
```

### 3. Feature Engineering

Feature engineering adalah proses menciptakan fitur-fitur baru dari data yang sudah ada untuk meningkatkan kemampuan model dalam menangkap pola-pola kompleks dalam data.

#### Alasan melakukan feature engineering

- Fitur-fitur baru yang relevan dapat membantu model dalam membuat prediksi yang lebih akurat.
- Fitur interaksi dapat membantu model dalam menangkap hubungan non-linear antar fitur.

**Implementasi Kode**

```python
# Membuat fitur interaksi antara 'age' dan 'cholesterol'
data_encoded['age_cholesterol'] = data_encoded['age'] * data_encoded['cholesterol']
```

**Hasil**

Dengan menambahkan fitur interaksi `age_cholesterol`, model diharapkan dapat menangkap hubungan kompleks antara umur pasien dan kadar kolesterol mereka, yang dapat berpengaruh signifikan terhadap risiko penyakit jantung.


### 4. Membagi Dataset menjadi Data Training dan Testing

Dataset yang telah dipersiapkan dibagi menjadi dua subset: data training dan data testing dengan proporsi 80:20. Pembagian ini dilakukan untuk melatih model pada data yang besar dan menguji performa model pada data yang belum pernah dilihat sebelumnya.

```python
from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Pembagian ini memastikan bahwa model dapat belajar dari data yang representatif dan dievaluasi secara objektif pada data yang berbeda.

## Modeling

Pada tahap ini, model machine learning dibangun dan dilatih untuk mengklasifikasikan risiko penyakit jantung berdasarkan data yang telah dipersiapkan sebelumnya. Dua algoritma machine learning yang dipilih untuk proyek ini adalah **Logistic Regression** dan **Decision Tree**. Pemilihan kedua algoritma ini didasarkan pada kemampuan mereka dalam menangani masalah klasifikasi biner serta interpretabilitas yang baik, yang sangat penting dalam konteks medis.

### Pemilihan Algoritma

1. **Logistic Regression**  
   Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memodelkan probabilitas kelas target berdasarkan satu atau lebih fitur input. Algoritma ini cocok untuk masalah klasifikasi biner dan memberikan interpretasi koefisien yang jelas mengenai kontribusi setiap fitur terhadap prediksi.  

   **Kelebihan:**  
   - Mudah diimplementasikan dan dipahami.
   - Memberikan probabilitas prediksi yang dapat diinterpretasikan.
   - Efisien untuk dataset yang besar.  

   **Kekurangan:**  
   - Asumsi linearitas antara fitur dan log-odds.
   - Kurang efektif untuk menangkap hubungan non-linear antar fitur.

2. **Decision Tree**  
   Decision Tree adalah algoritma klasifikasi yang membangun model berupa pohon keputusan berdasarkan fitur-fitur input. Model ini memecah data secara rekursif berdasarkan fitur yang paling informatif.  

   **Kelebihan:**
   - Mudah diinterpretasikan dan divisualisasikan.
   - Dapat menangkap hubungan non-linear antar fitur.
   - Tidak memerlukan feature scaling.  

   **Kekurangan:**
   - Rentan terhadap overfitting, terutama pada pohon yang dalam.
   - Kurang stabil karena perubahan kecil pada data dapat menghasilkan pohon yang berbeda.

### Melatih Model Logistic Regression

Model Logistic Regression diinisialisasi dan dilatih menggunakan data training.
Parameter yang digunakan adalah `max_iter=1000` dan `random_state=42`. Ini memastikan bahwa model dilatih hingga konvergensi dan hasilnya dapat direproduksi.
Pada fungsi `fit()`, model dilatih menggunakan data training `X_train` dan label `y_train`.

```python
from sklearn.linear_model import LogisticRegression

# Inisialisasi model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Training model
log_reg.fit(X_train, y_train)
```

Logistic Regression dipilih karena kemampuannya dalam memberikan interpretasi koefisien yang jelas dan efisiensi dalam pelatihan.  

### Melatih Model Decision Tree

Model Decision Tree diinisialisasi dan dilatih menggunakan data training.
Parameter yang digunakan adalah `random_state=42` untuk memastikan hasil yang konsisten.
Pada fungsi `fit()`, model dilatih menggunakan data training `X_train` dan label `y_train`.

```python
from sklearn.tree import DecisionTreeClassifier

# Inisialisasi model
dt = DecisionTreeClassifier(random_state=42)

# Training model
dt.fit(X_train, y_train)
```

Decision Tree dipilih karena kemampuannya dalam menangkap hubungan non-linear antar fitur dan kemudahan dalam interpretasi.

### Hyperparameter Tuning

Untuk meningkatkan performa kedua model, dilakukan optimasi hyperparameter menggunakan teknik **Grid Search** dengan cross-validation.
Cross-validation diimplementasikan melalui **GridSearchCV** dengan parameter `cv=5`, yang berarti **5-fold cross-validation**.

**a. GridSearchCV dengan Cross-Validation:**
   - GridSearchCV adalah metode yang digunakan untuk mencari kombinasi hyperparameter terbaik dari sekumpulan hyperparameter yang telah ditentukan.
   - **Proses:**
     - Dataset dibagi menjadi 5 fold.
     - Untuk setiap kombinasi hyperparameter, model dilatih pada 4 fold dan diuji pada fold ke-5.
     - Proses ini diulang hingga setiap fold telah digunakan sebagai data testing.
     - Performanya diukur menggunakan metrik evaluasi yang ditentukan (misalnya, ROC-AUC).
     - Rata-rata performa dari 5 fold digunakan untuk menentukan kombinasi hyperparameter terbaik.

**b. Keuntungan Penggunaan GridSearchCV dengan Cross-Validation:**
   - GridSearchCV secara sistematis mengevaluasi semua kombinasi hyperparameter yang mungkin, memastikan bahwa kita menemukan kombinasi yang memberikan performa terbaik.
   - Dengan menggunakan cross-validation, kita mendapatkan estimasi performa yang lebih andal karena model diuji pada berbagai subset data.

**1. Hyperparameter Tuning untuk Logistic Regression**  

```python
from sklearn.model_selection import GridSearchCV

# Definisikan grid parameter
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

# Inisialisasi GridSearchCV
grid_log_reg = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=5, scoring='roc_auc')

# Training GridSearchCV
grid_log_reg.fit(X_train, y_train)
```

**Penjelasan parameter:**
**a. `param_grid_lr`**  
   - **`C`**: Parameter regularisasi dalam Logistic Regression yang mengontrol kekuatan regularisasi. Nilai yang lebih kecil menunjukkan regularisasi lebih kuat, sedangkan nilai yang lebih besar menunjukkan regularisasi lebih lemah.
      - **[0.01, 0.1, 1, 10, 100]**: Berbagai nilai regularisasi yang akan diuji.
   - **`solver`**: Algoritma yang digunakan untuk optimisasi Logistic Regression.
      - **`liblinear`**: Solver untuk dataset kecil atau sparsitas tinggi, menggunakan metode coordinate descent.
      - **`lbfgs`**: Solver berbasis metode optimasi quasi-Newton, lebih cocok untuk dataset dengan banyak fitur.

**b. `GridSearchCV`**  
   - **`LogisticRegression(max_iter=1000, random_state=42)`**: Model Logistic Regression yang akan dioptimasi.
      - **`max_iter=1000`**: Menentukan jumlah iterasi maksimum sebesar 1000 selama proses optimasi.
      - **`random_state=42`**: Untuk memastikan hasil yang konsisten dan dapat direproduksi.
   - **`param_grid_lr`**: Grid parameter yang didefinisikan di atas, yaitu `C` dan `solver`, untuk mencoba semua kombinasi parameter yang memungkinkan.
   - **`cv=5`**: Cross-validation dengan **5 folds**. Dataset dibagi menjadi 5 bagian, di mana 4 bagian digunakan untuk pelatihan dan 1 bagian untuk validasi, dilakukan secara bergantian.
   - **`scoring='roc_auc'`**: Metrik evaluasi yang digunakan adalah **ROC-AUC**, untuk mengukur kemampuan model dalam membedakan antara kelas positif dan negatif.

**c. `grid_log_reg.fit(X_train, y_train)`**  
   - Proses pelatihan GridSearchCV pada data pelatihan (**X_train, y_train**).
   - Model akan mencoba setiap kombinasi parameter dari `param_grid_lr` dengan 5-fold cross-validation.
   - Hasil akhir berupa model Logistic Regression dengan kombinasi parameter terbaik berdasarkan metrik **ROC-AUC**.

**2. Hyperparameter Tuning untuk Decision Tree**

```python
# Definisikan grid parameter
param_grid_dt = {
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi GridSearchCV
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='roc_auc')

# Training GridSearchCV
grid_dt.fit(X_train, y_train)
```

**Penjelasan parameter:**
Berikut adalah penjelasan dari setiap parameter dalam kode tersebut:

**a. `param_grid_dt`**  
   - **`max_depth`**: 
     - Menentukan kedalaman maksimum pohon keputusan.
     - **[None, 5, 10, 20, 30]**: 
       - `None` berarti pohon dapat berkembang hingga semua daun hanya memiliki satu sampel atau mencapai kedalaman maksimum tanpa batas.
       - Nilai angka seperti 5, 10, 20, 30 membatasi kedalaman pohon untuk mencegah overfitting.
   - **`min_samples_split`**: 
     - Jumlah minimum sampel yang diperlukan untuk membagi sebuah node.
     - **[2, 5, 10]**:
       - Nilai lebih kecil memungkinkan pembagian lebih sering (berpotensi overfitting).
       - Nilai lebih besar mengurangi pembagian, meningkatkan generalisasi.
   - **`min_samples_leaf`**: 
     - Jumlah minimum sampel yang diperlukan dalam satu daun (leaf node).
     - **[1, 2, 4]**:
       - Nilai 1 berarti daun dapat memiliki satu sampel, memungkinkan pembagian mendetail.
       - Nilai lebih tinggi membantu mencegah daun terlalu kecil, meningkatkan generalisasi.

**b. `GridSearchCV`**  
   - **`DecisionTreeClassifier(random_state=42)`**: 
     - Model Decision Tree yang akan dioptimasi.
     - **`random_state=42`**: Mengatur seed random untuk hasil yang dapat direproduksi.
   - **`param_grid_dt`**: 
     - Grid parameter yang didefinisikan di atas, yaitu kombinasi dari `max_depth`, `min_samples_split`, dan `min_samples_leaf`, untuk mencoba setiap kombinasi yang memungkinkan.
   - **`cv=5`**:
     - Cross-validation dengan **5 folds**.
     - Dataset dibagi menjadi 5 bagian; pada setiap iterasi, 4 bagian digunakan untuk pelatihan dan 1 bagian untuk validasi, dilakukan secara bergantian.
   - **`scoring='roc_auc'`**: 
     - Metrik evaluasi yang digunakan adalah **ROC-AUC** untuk mengukur kemampuan model dalam membedakan antara kelas positif dan negatif.

**c. `grid_dt.fit(X_train, y_train)`**  
   - Proses pelatihan **GridSearchCV** pada data pelatihan (**X_train, y_train**).
   - Semua kombinasi parameter dari `param_grid_dt` diuji menggunakan 5-fold cross-validation.
   - Hasil akhir berupa model Decision Tree dengan kombinasi parameter terbaik berdasarkan metrik **ROC-AUC**.

## Evaluasi
### 1. Metrik

**a. Accuracy (Akurasi)**  
**Rumus:**  
$$\text{Accuracy} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Jumlah Prediksi}}$$

Accuracy mengukur proporsi prediksi yang benar dari keseluruhan data yang diuji. Metrik ini memberikan gambaran umum tentang performa model, namun dapat menyesatkan jika dataset memiliki ketidakseimbangan kelas (imbalanced classes).

**Kelebihan:**
- Mudah dipahami dan dihitung.
- Memberikan gambaran umum performa model.

**Kekurangan:**
- Tidak efektif pada dataset dengan distribusi kelas yang tidak seimbang.
- Tidak memberikan informasi tentang jenis kesalahan yang dilakukan model.

**b. Precision (Presisi)**

**Rumus:**
$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

Precision mengukur proporsi prediksi positif yang benar-benar positif. Metrik ini penting ketika biaya dari false positive tinggi, seperti dalam deteksi penyakit di mana diagnosis yang salah dapat menyebabkan kecemasan pasien dan biaya tambahan.

**Kelebihan:**
- Menilai ketepatan prediksi positif.
- Berguna dalam konteks di mana false positive harus diminimalisir.

**Kekurangan:**
- Tidak mempertimbangkan false negative.
- Dapat rendah jika model sering memprediksi positif secara berlebihan.

**c. Recall (Sensitivitas)**

**Rumus:**
$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

Recall mengukur proporsi aktual positif yang berhasil diprediksi oleh model. Metrik ini penting dalam konteks medis karena false negative dapat berarti pasien yang sebenarnya berisiko tidak terdeteksi, sehingga tidak mendapatkan perawatan yang diperlukan.

**Kelebihan:**
- Menilai kemampuan model dalam mendeteksi semua kasus positif.
- Berguna dalam konteks di mana false negative harus diminimalisir.

**Kekurangan:**
- Tidak mempertimbangkan false positive.
- Dapat tinggi jika model cenderung memprediksi positif secara berlebihan.

**d. F1-Score**

#### **Rumus:**
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1-Score adalah harmonic mean dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut. Metrik ini berguna ketika terdapat kebutuhan untuk menyeimbangkan antara precision dan recall, terutama pada dataset dengan distribusi kelas yang tidak seimbang.

**Kelebihan:**
- Menggabungkan precision dan recall menjadi satu metrik.
- Memberikan gambaran yang lebih seimbang tentang performa model dibandingkan hanya menggunakan accuracy.

**Kekurangan:**
- Tidak memberikan informasi tentang true negatives.
- Sulit diinterpretasikan secara intuitif dibandingkan dengan accuracy.

**e. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**

![ROC Curve](/img/5.1.png)

ROC-AUC mengukur kemampuan model dalam membedakan antara kelas positif dan negatif. Nilai AUC berkisar antara 0.5 (model tidak lebih baik dari tebakan acak) hingga 1.0 (model sempurna). Metrik ini berguna untuk menilai performa model secara keseluruhan tanpa tergantung pada threshold tertentu.

**Kelebihan:**
- Menilai kemampuan model dalam membedakan antara kelas positif dan negatif.
- Tidak tergantung pada threshold tertentu.
- Berguna untuk membandingkan performa model yang berbeda.

**Kekurangan:**
- Interpretasi AUC bisa kurang intuitif dibandingkan metrik lain seperti accuracy atau F1-score.
- Tidak memberikan informasi tentang nilai prediksi aktual atau threshold yang optimal.

### 2. Evaluasi Awal Model

Setelah melatih kedua model, kita melakukan prediksi pada data testing dan mengevaluasi performanya menggunakan berbagai metrik evaluasi.

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Prediksi dengan Logistic Regression
y_pred_lr = log_reg.predict(X_test)
y_pred_proba_lr = log_reg.predict_proba(X_test)[:,1]

# Prediksi dengan Decision Tree
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:,1]
```

#### Model Logistic Regression

Hasil evaluasi awal model pada model logistic regression adalah sebagai berikut.

![Hasil evaluasi awal model logistic regression](/img/5.2.png)

Berikut adalah insight dari **ROC Curve** untuk model **Logistic Regression**:

**a. Area Under Curve (AUC = 0.93):**
   - AUC yang mendekati 1 (0.93) menunjukkan bahwa model Logistic Regression memiliki performa yang sangat baik dalam membedakan antara kelas positif (berisiko penyakit jantung) dan kelas negatif (tidak berisiko).
   - Nilai AUC sebesar 0.93 mengindikasikan bahwa model memiliki tingkat kemampuan prediksi yang tinggi, dengan peluang 93% untuk membedakan pasien berisiko dan tidak berisiko secara benar.

**b. True Positive Rate (TPR) vs. False Positive Rate (FPR):**
   - ROC Curve memperlihatkan TPR (y-axis) yang tinggi pada FPR yang rendah di awal kurva, menunjukkan bahwa model mampu mendeteksi banyak kasus berisiko penyakit jantung dengan sedikit kesalahan (false positives).
   - Kurva yang cepat naik mendekati sudut kiri atas menunjukkan sensitivitas tinggi pada FPR rendah, yang penting dalam konteks kesehatan karena model ini dapat membantu mengidentifikasi pasien berisiko dengan akurasi tinggi sambil meminimalkan kesalahan deteksi pada pasien yang tidak berisiko.

**c. Keseimbangan antara Sensitivitas dan Spesifisitas:**
   - ROC Curve ini menunjukkan keseimbangan yang baik antara sensitivitas (kemampuan model untuk mendeteksi kasus positif) dan spesifisitas (kemampuan model untuk mendeteksi kasus negatif).
   - Dalam praktiknya, ini menunjukkan bahwa model Logistic Regression cocok untuk deteksi dini risiko penyakit jantung, karena memiliki kemampuan yang baik dalam mengidentifikasi pasien berisiko tanpa terlalu banyak menghasilkan alarm palsu.

Sedangkan, hasil untuk classification report adalah sebagai berikut.

```text
Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.81      0.84       112
           1       0.84      0.89      0.86       126

    accuracy                           0.85       238
   macro avg       0.85      0.85      0.85       238
weighted avg       0.85      0.85      0.85       238

ROC-AUC: 0.9309098639455782
```

Berikut adalah analisis terhadap hasil model baseline **Logistic Regression** berdasarkan classification report dan nilai ROC-AUC:

**a. Precision**
   - **Kelas 0 (No Disease):** Precision adalah **0.87**, yang berarti dari semua prediksi "tidak berisiko penyakit jantung," 87% adalah benar. Model cukup baik dalam memastikan bahwa pasien yang diprediksi tidak berisiko benar-benar sehat.
   - **Kelas 1 (Disease):** Precision adalah **0.84**, yang menunjukkan bahwa model dapat secara akurat memprediksi pasien berisiko penyakit jantung, meskipun ada beberapa false positives.

**b. Recall**
   - **Kelas 0 (No Disease):** Recall adalah **0.81**, yang berarti model mendeteksi 81% dari pasien yang benar-benar tidak berisiko. Meskipun cukup baik, ada peluang untuk meningkatkan recall pada kelas ini.
   - **Kelas 1 (Disease):** Recall adalah **0.89**, yang berarti model dapat mendeteksi 89% dari pasien yang benar-benar berisiko. Ini sangat penting dalam konteks medis untuk mengurangi kemungkinan pasien berisiko tidak terdeteksi (false negatives).

**c. F1-Score**
   - **Kelas 0 (No Disease):** F1-Score adalah **0.84**, yang menunjukkan keseimbangan yang baik antara precision dan recall untuk kelas ini.
   - **Kelas 1 (Disease):** F1-Score adalah **0.86**, yang menunjukkan bahwa model memiliki performa lebih baik dalam mendeteksi pasien berisiko dibandingkan yang tidak berisiko.

**d. Accuracy**
   - Akurasi keseluruhan model adalah **85%**, menunjukkan bahwa model secara keseluruhan mampu memprediksi 85% dari semua kasus dengan benar. Ini merupakan peningkatan dibandingkan Decision Tree yang hanya memiliki akurasi 82%.

**e. ROC-AUC**
   - Nilai ROC-AUC adalah **0.93**, yang berarti model Logistic Regression memiliki kemampuan yang sangat baik dalam membedakan antara pasien yang berisiko dan tidak berisiko. Nilai ini jauh lebih baik dibandingkan Decision Tree (ROC-AUC: 0.83).

Model Logistic Regression memberikan performa yang sangat baik dengan akurasi **85%** dan ROC-AUC **0.93**. Model ini unggul dalam mendeteksi pasien berisiko (kelas 1) dengan recall yang tinggi (89%) serta memberikan keseimbangan yang baik antara precision dan recall. Logistic Regression adalah pilihan baseline yang solid dan dapat digunakan untuk kasus prediksi risiko penyakit jantung dengan sedikit perbaikan pada recall kelas 0 dan pengayaan fitur.

#### Model Decision Tree

Hasil evaluasi awal model pada model decision tree adalah sebagai berikut.

![Hasil evaluasi awal model decision tree](/img/5.3.png)

Berikut adalah insight dari **ROC Curve** untuk model **Decision Tree**:

**a. Area Under Curve (AUC = 0.83):**
   - AUC sebesar 0.83 menunjukkan bahwa model Decision Tree memiliki performa yang lumayan dalam membedakan antara kelas positif (berisiko penyakit jantung) dan kelas negatif (tidak berisiko), namun tidak sebaik model Logistic Regression yang memiliki AUC sebesar 0.93.

**b. True Positive Rate (TPR) vs. False Positive Rate (FPR):**
   - ROC Curve menunjukkan bahwa model ini berhasil mencapai TPR tinggi di awal kurva pada FPR yang rendah. Namun, kurvanya lebih mendekati garis diagonal setelah titik awal, yang menunjukkan bahwa model ini cenderung lebih sederhana dan mungkin kehilangan beberapa kasus positif yang lebih sulit dideteksi.
   - Peningkatan TPR yang tidak sehalus Logistic Regression menunjukkan bahwa Decision Tree mungkin kurang optimal dalam menangani kompleksitas dalam data.

**c. Keseimbangan Sensitivitas dan Spesifisitas:**
   - Meskipun Decision Tree memiliki kemampuan dasar untuk membedakan antara pasien berisiko dan tidak berisiko, model ini kurang tepat dalam menangkap semua pola dalam data dibandingkan model yang lebih kompleks seperti Logistic Regression.
   - Dalam konteks penyakit jantung, di mana akurasi deteksi sangat penting, hasil ini menunjukkan bahwa Decision Tree mungkin perlu ditingkatkan melalui teknik seperti pruning atau dengan menggunakan model ensemble seperti Random Forest atau Gradient Boosting.

Sedangkan, hasil untuk classification report adalah sebagai berikut.

```text
Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.86      0.82       112
           1       0.86      0.79      0.83       126

    accuracy                           0.82       238
   macro avg       0.82      0.83      0.82       238
weighted avg       0.83      0.82      0.82       238

ROC-AUC: 0.8253968253968254
```

Berikut adalah analisis terhadap hasil model baseline **Decision Tree** berdasarkan classification report dan nilai ROC-AUC:

**a. Precision**
   - **Kelas 0 (No Disease):** Precision adalah **0.79**, yang berarti dari semua prediksi "tidak berisiko penyakit jantung," 79% adalah benar. Ini menunjukkan bahwa model memiliki kecenderungan untuk menghasilkan false positives, di mana pasien diprediksi tidak memiliki risiko padahal sebenarnya berisiko.
   - **Kelas 1 (Disease):** Precision adalah **0.86**, yang menunjukkan bahwa model cukup akurat dalam memprediksi pasien yang berisiko penyakit jantung. Model lebih bisa diandalkan dalam mendeteksi kasus berisiko daripada yang tidak berisiko.

**b. Recall**
   - **Kelas 0 (No Disease):** Recall adalah **0.86**, yang menunjukkan bahwa model dapat mendeteksi 86% dari total pasien yang benar-benar tidak berisiko. Ini penting untuk memastikan model tidak mengabaikan banyak pasien sehat (false negatives rendah).
   - **Kelas 1 (Disease):** Recall adalah **0.79**, yang berarti model mendeteksi 79% pasien yang benar-benar berisiko. Namun, ada sekitar 21% pasien berisiko yang tidak terdeteksi oleh model, yang dapat menjadi perhatian dalam konteks medis.

**c. F1-Score**
   - F1-Score adalah kombinasi dari precision dan recall, memberikan gambaran umum tentang keseimbangan antara keduanya.
   - **Kelas 0:** F1-Score adalah **0.82**, yang menunjukkan performa yang baik dalam mendeteksi pasien tanpa risiko.
   - **Kelas 1:** F1-Score adalah **0.83**, yang berarti model lebih baik dalam mendeteksi pasien berisiko dibandingkan yang tidak berisiko.

**d. Accuracy**
   - Akurasi keseluruhan model adalah **82%**, yang berarti 82% dari total prediksi model adalah benar. Namun, akurasi ini bisa menutupi ketidakseimbangan atau performa buruk pada salah satu kelas.

**e. ROC-AUC**
   - Nilai ROC-AUC adalah **0.825**, menunjukkan bahwa model memiliki kemampuan yang cukup baik untuk membedakan antara pasien yang berisiko dan tidak berisiko. Semakin dekat nilai ini ke 1, semakin baik performa model dalam membedakan kedua kelas.

Model baseline Decision Tree menunjukkan performa yang cukup baik sebagai awal, dengan akurasi **82%** dan ROC-AUC **0.83**. Namun, terdapat ruang untuk perbaikan, khususnya dalam recall untuk mendeteksi pasien berisiko (kelas 1), yang sangat penting dalam konteks kesehatan.
Dari evaluasi awal, terlihat bahwa Decision Tree menunjukkan bahwa model ini memiliki kemampuan klasifikasi yang cukup baik tetapi tidak seoptimal Logistic Regression pada dataset ini.

### 3. Evaluasi Model Setelah Hyperparameter Tuning

Setelah melakukan hyperparameter tuning, model yang telah dioptimasi dievaluasi kembali menggunakan data testing.

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Prediksi dengan Logistic Regression
y_pred_lr_best = grid_log_reg.predict(X_test)
y_pred_proba_lr_best = grid_log_reg.predict_proba(X_test)[:,1]

# Prediksi dengan Decision Tree
y_pred_dt_best = grid_dt.predict(X_test)
y_pred_proba_dt_best = grid_dt.predict_proba(X_test)[:,1]
```

#### Model Logistic Regression

Berikut merupakan hasil evaluasi model logistic regression setelah hyperparameter tuning.

![Hasil evaluasi model logistic regression setelah hyperparameter tuning](/img/5.4.png)

Hasil ROC curve setelah hyperparameter tuning menunjukkan bahwa nilai AUC tetap 0.93, sama seperti model baseline. Ini menunjukkan bahwa hyperparameter tuning tidak memberikan peningkatan performa yang signifikan untuk kemampuan model membedakan antara kelas. Namun, tuning dapat membantu meningkatkan stabilitas model dan memastikan bahwa model beroperasi dengan parameter optimal. Logistic Regression tetap menjadi model yang sangat baik untuk tugas ini.

Sedangkan, hasil untuk classification report adalah sebagai berikut.

```text
Best Parameters for Logistic Regression: {'C': 0.1, 'solver': 'lbfgs'}
Best ROC-AUC Score: 0.912373340255374
Logistic Regression Best Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.82      0.84       112
           1       0.85      0.88      0.86       126

    accuracy                           0.85       238
   macro avg       0.85      0.85      0.85       238
weighted avg       0.85      0.85      0.85       238

Logistic Regression Best ROC-AUC: 0.930484693877551
```

Setelah melakukan hyperparameter tuning, model Logistic Regression dengan parameter terbaik (C=0.1, solver='lbfgs') memberikan performa yang serupa dengan baseline model. Akurasi tetap 85%, dengan precision, recall, dan f1-score yang merata di kedua kelas. Nilai ROC-AUC sedikit menurun dari baseline pada proses tuning (0.912 selama tuning dan 0.930 pada testing), menunjukkan bahwa tuning tidak memberikan peningkatan yang signifikan dalam kemampuan prediktif model, tetapi tetap optimal dan stabil.

#### Model Decision Tree

Berikut merupakan hasil evaluasi model decision tree setelah hyperparameter tuning.

![Hasil evaluasi model decision tree setelah hyperparameter tuning](/img/5.5.png)

ROC Curve untuk Decision Tree setelah hyperparameter tuning menunjukkan peningkatan Area Under Curve (AUC) menjadi 0.88, dibandingkan baseline sebelumnya yang 0.83. Hal ini menunjukkan bahwa tuning membantu meningkatkan kemampuan model dalam membedakan antara kelas positif dan negatif. Namun, performanya masih berada di bawah Logistic Regression, yang menunjukkan AUC 0.93.

Sedangkan, hasil untuk classification report adalah sebagai berikut.

```text
Best Parameters for Decision Tree: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10}
Best ROC-AUC Score: 0.8901675121444729
Decision Tree Best Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.84      0.82       112
           1       0.85      0.81      0.83       126

    accuracy                           0.82       238
   macro avg       0.82      0.82      0.82       238
weighted avg       0.82      0.82      0.82       238

Decision Tree Best ROC-AUC: 0.8766652494331065
```

Hyperparameter tuning pada Decision Tree menghasilkan parameter terbaik: `max_depth=None`, `min_samples_leaf=1`, dan `min_samples_split=10`. Model menunjukkan peningkatan **ROC-AUC** menjadi **0.88**, dengan akurasi keseluruhan **82%**. Recall untuk kelas negatif dan positif relatif seimbang (**0.84** dan **0.81**), menunjukkan bahwa model dapat menangani kedua kelas dengan cukup baik, meskipun masih berada di bawah performa Logistic Regression.

### 4. Pemilihan Model Terbaik

Setelah evaluasi, performa kedua model dibandingkan berdasarkan metrik evaluasi yang telah ditentukan. Model dengan skor ROC-AUC tertinggi dianggap sebagai model terbaik.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Mengumpulkan metrik evaluasi untuk kedua model
metrics = {
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr_best), accuracy_score(y_test, y_pred_dt_best)],
    'Precision': [precision_score(y_test, y_pred_lr_best), precision_score(y_test, y_pred_dt_best)],
    'Recall': [recall_score(y_test, y_pred_lr_best), recall_score(y_test, y_pred_dt_best)],
    'F1-Score': [f1_score(y_test, y_pred_lr_best), f1_score(y_test, y_pred_dt_best)],
    'ROC-AUC': [roc_auc_score(y_test, y_pred_proba_lr_best), roc_auc_score(y_test, y_pred_proba_dt_best)]
}

# Membuat DataFrame dari metrik
metrics_df = pd.DataFrame(metrics)

# Menampilkan DataFrame metrik
print(metrics_df)
```

#### Hasil:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.852941     | 0.847328      | 0.880952   | 0.864865     | 0.930485    |
| Decision Tree       | 0.823529     | 0.850000      | 0.809524   | 0.829268     | 0.876665    |

Dari hasil yang muncul, tampak bahwa model **Logistic Regression** memiliki performa yang lebih baik dibandingkan **Decision Tree** berdasarkan metrik evaluasi yang telah ditentukan. Model Logistic Regression memiliki nilai **ROC-AUC** yang lebih tinggi (0.93) dibandingkan Decision Tree (0.88), menunjukkan kemampuan yang lebih baik dalam membedakan antara pasien berisiko dan tidak berisiko. Selain itu, Logistic Regression juga memiliki nilai **Accuracy**, **Precision**, **Recall**, dan **F1-Score** yang lebih baik dibandingkan Decision Tree.

### 5. Implementasi dan Penyimpanan Model Terbaik

Model terbaik (Logistic Regression) disimpan menggunakan library **joblib** agar dapat digunakan kembali tanpa perlu melatih ulang.

```python
import joblib

# Menyimpan model terbaik
joblib.dump(grid_log_reg.best_estimator_, 'logistic_regression_best_model.pkl')
print("Model Logistic Regression terbaik telah disimpan sebagai 'logistic_regression_best_model.pkl'.")
```

### 8. Memuat dan Menguji Model Terbaik

```python
# Memuat model terbaik
best_model_loaded = joblib.load('logistic_regression_best_model.pkl')
print("Model Logistic Regression terbaik berhasil dimuat.")

# Melakukan prediksi dengan model yang dimuat
y_pred_loaded = best_model_loaded.predict(X_test)
y_pred_proba_loaded = best_model_loaded.predict_proba(X_test)[:,1]
```

Proses ini dilakukan untuk memastikan bahwa model yang disimpan dapat dimuat dan bekerja dengan baik di masa depan, memungkinkan penggunaan ulang model tanpa perlu pelatihan ulang.

### 9. Visualisasi Perbandingan Metrik

Untuk memudahkan perbandingan performa kedua model, berikut adalah visualisasi metrik evaluasi dalam bentuk tabel dan grafik.

**Perbandingan Kedua Model dengan Diagram Batang**

Berikut merupakan perbandingan metrik evaluasi antara model Logistic Regression dan Decision Tree dalam bentuk diagram batang.

![Perbandingan metrik evaluasi antara model Logistic Regression dan Decision Tree](/img/5.6.png)

Visualisasi ini diharapkan dapat memberikan gambaran yang jelas tentang perbedaan performa antara kedua model.

**Perbandingan ROC Curve antara Kedua Model**

![Perbandingan ROC Curve antara model Logistic Regression dan Decision Tree](/img/5.7.png)

Dari grafik di atas, terlihat bahwa model Logistic Regression memiliki kurva ROC yang lebih mendekati sudut kiri atas, menunjukkan kemampuan yang lebih baik dalam membedakan antara pasien berisiko dan tidak berisiko dibandingkan Decision Tree. Meskipun Logistic Regression tampak lebih unggul, selisih antara keduanya tidak jauh berbeda.

**Confusion Matrix untuk Kedua Model**

![Confusion Matrix untuk model Logistic Regression](/img/5.8.png)

Berikut adalah insight dari **Confusion Matrix** untuk model **Logistic Regression** dan **Decision Tree**:

**a. Logistic Regression:**
   - **True Positives (TP)**: 112 — Kasus pasien dengan penyakit jantung yang terdeteksi dengan benar oleh model.
   - **True Negatives (TN)**: 91 — Kasus pasien tanpa penyakit jantung yang diklasifikasikan dengan benar sebagai "No Disease".
   - **False Positives (FP)**: 21 — Kasus pasien yang sebenarnya tidak memiliki penyakit jantung tetapi diklasifikasikan salah sebagai "Disease".
   - **False Negatives (FN)**: 14 — Kasus pasien yang sebenarnya memiliki penyakit jantung tetapi tidak terdeteksi oleh model.

   Logistic Regression menunjukkan performa yang kuat, dengan jumlah FN yang lebih sedikit (14), yang berarti model ini lebih andal dalam mendeteksi pasien yang benar-benar berisiko. Meskipun ada beberapa kesalahan klasifikasi (FP dan FN), model ini secara keseluruhan memiliki keseimbangan yang baik antara mendeteksi penyakit dan menghindari kesalahan positif.

**b. Decision Tree:**
   - **True Positives (TP)**: 100 — Model mendeteksi penyakit jantung dengan benar pada 100 pasien.
   - **True Negatives (TN)**: 96 — Model berhasil mengidentifikasi dengan benar pasien tanpa penyakit jantung.
   - **False Positives (FP)**: 16 — Model mengklasifikasikan beberapa pasien tanpa penyakit sebagai "Disease".
   - **False Negatives (FN)**: 26 — Kasus penyakit jantung yang tidak terdeteksi oleh model.

   Decision Tree memiliki jumlah FN yang lebih tinggi (26), menunjukkan bahwa model ini lebih sering gagal mendeteksi kasus penyakit jantung dibandingkan Logistic Regression. Jumlah TP yang lebih rendah juga menunjukkan bahwa model ini tidak sebaik Logistic Regression dalam mengidentifikasi pasien yang berisiko.

**Tambahan: Feature Importance pada Decision Tree**

Selain evaluasi performa model, kita juga dapat melihat **Feature Importance** pada model Decision Tree untuk memahami kontribusi relatif dari setiap fitur dalam membuat prediksi.

![Feature Importance pada model Decision Tree](/img/5.9.png)  

Berikut adalah insight dari grafik **Feature Importance** pada model **Decision Tree**:

**a. Fitur Paling Penting**
   - **ST slope_1** merupakan fitur paling penting dengan skor kepentingan tertinggi. Hal ini menunjukkan bahwa tipe ST slope ini memiliki pengaruh besar dalam menentukan apakah seseorang berisiko penyakit jantung atau tidak. Perubahan pada slope segmen ST biasanya terkait dengan kondisi jantung, sehingga fitur ini relevan untuk deteksi penyakit jantung.
   - **Chest pain type_4** (nyeri dada asimptomatik) juga memiliki pengaruh besar. Nyeri dada tanpa gejala sering kali menjadi tanda bahaya dalam konteks medis, terutama untuk penyakit jantung.

**b. Fitur Lain yang Signifikan**
   - **Oldpeak**, **max heart rate**, dan **resting bp s** juga memiliki pengaruh yang signifikan terhadap klasifikasi risiko penyakit jantung. Ini masuk akal karena nilai oldpeak menunjukkan depresi ST, yang dapat menjadi tanda abnormalitas jantung. Max heart rate dan resting bp s (tekanan darah saat istirahat) adalah indikator penting untuk menilai kesehatan jantung secara keseluruhan.
   - **Cholesterol** juga cukup penting dalam model ini, mengingat kadar kolesterol tinggi dapat meningkatkan risiko penyakit jantung.

**c. Fitur Demografis**
   - **Sex** dan **age** juga memiliki kontribusi, meskipun lebih rendah dibandingkan fitur kesehatan lainnya. Hal ini masuk akal karena risiko penyakit jantung sering kali dipengaruhi oleh faktor usia dan jenis kelamin, di mana pria dan orang yang lebih tua cenderung memiliki risiko yang lebih tinggi.

**d. Interaksi Antar-Fitur**
   - Fitur baru yang merupakan interaksi antara **age** dan **cholesterol** (age_cholesterol) juga memiliki kontribusi, meskipun tidak terlalu besar. Ini mungkin menunjukkan bahwa kolesterol yang tinggi pada kelompok usia tertentu meningkatkan risiko lebih lanjut.

**d. Fitur dengan Pengaruh Rendah**
   - Beberapa fitur, seperti **ST slope_3**, **chest pain type_2**, dan **resting ecg_1**, memiliki skor kepentingan yang rendah. Hal ini menunjukkan bahwa model Decision Tree menganggap fitur-fitur ini kurang relevan untuk prediksi risiko penyakit jantung dalam dataset ini.

## Penutup

### Kesimpulan

Proyek ini berhasil mengembangkan model machine learning yang efektif untuk mengklasifikasikan risiko penyakit jantung berdasarkan data kesehatan dan demografis pasien. Dalam proses persiapan data, penerapan **One-Hot Encoding** untuk fitur kategorikal dan **StandardScaler** untuk fitur numerik memastikan bahwa data berada dalam format yang sesuai dan memiliki skala yang seimbang. Selain itu, penambahan fitur interaksi `age_cholesterol` memungkinkan model untuk menangkap hubungan kompleks antar fitur, yang berkontribusi pada peningkatan performa model secara keseluruhan.

Dalam tahap evaluasi, **Logistic Regression** menunjukkan performa yang lebih unggul dengan akurasi sebesar **85%** dan skor **ROC-AUC** sebesar **0.93**, dibandingkan dengan **Decision Tree** yang mencapai akurasi **82%** dan ROC-AUC **0.83**. Hasil ini menunjukkan bahwa Logistic Regression tidak hanya unggul dalam hal akurasi tetapi juga memberikan keseimbangan yang baik antara precision dan recall, yang sangat penting dalam konteks medis untuk memastikan deteksi risiko penyakit jantung yang akurat dan mengurangi kesalahan diagnosis. Meskipun optimasi hyperparameter melalui **Grid Search** dengan cross-validation berhasil meningkatkan performa **Decision Tree** hingga ROC-AUC sebesar **0.88**, **Logistic Regression** tetap mempertahankan performa terbaiknya setelah tuning, menunjukkan bahwa model ini berada pada konfigurasi optimal yang mampu memberikan performa stabil.

Proyek ini secara efektif menjawab semua problem statements yang diajukan. Pertama, model berhasil mengklasifikasikan pasien sebagai berisiko atau tidak berisiko penyakit jantung berdasarkan data kesehatan dan demografis yang tersedia. Kedua, melalui evaluasi yang komprehensif, proyek ini mengidentifikasi bahwa **Logistic Regression** adalah algoritma yang paling efektif dibandingkan **Decision Tree** dalam konteks dataset ini. Ketiga, peningkatan akurasi model dicapai melalui optimasi parameter dan teknik pemodelan lanjutan. Hal ini memastikan bahwa model dapat memberikan prediksi yang lebih akurat. Keempat, model yang dibangun tidak hanya akurat tetapi juga interpretatif,dapat mendukung pengambilan keputusan medis yang lebih baik dengan memberikan wawasan tentang kontribusi masing-masing fitur terhadap risiko penyakit jantung.

Dengan memilih algoritma yang tepat dan melakukan optimasi yang diperlukan, proyek ini mencapai tujuan utamanya yaitu membangun model klasifikasi yang akurat dan dapat diandalkan. **Logistic Regression** yang dipilih sebagai model terbaik memberikan interpretabilitas yang tinggi, memungkinkan tenaga medis untuk memahami faktor-faktor yang berkontribusi terhadap risiko penyakit jantung secara lebih mendalam. Hal ini tidak hanya meningkatkan efisiensi deteksi dini tetapi juga memperkuat kepercayaan dalam penggunaan model sebagai alat bantu dalam diagnosis medis.

Secara keseluruhan, proyek ini berhasil mencapai semua goals yang diharapkan. Model yang dikembangkan tidak hanya memenuhi standar akurasi dan performa tetapi juga memberikan manfaat praktis dalam konteks klinis. Dengan implementasi model ini ke dalam sistem pendukung keputusan medis, diharapkan dapat meningkatkan kualitas hidup pasien melalui deteksi dini risiko penyakit jantung dan intervensi yang lebih tepat waktu. Meskipun terdapat beberapa keterbatasan, seperti ukuran dataset dan jumlah fitur yang digunakan, hasil yang diperoleh membuka peluang untuk penelitian lebih lanjut dan pengembangan aplikasi praktis yang dapat meningkatkan kualitas layanan kesehatan.

### Saran

Untuk penelitian selanjutnya, disarankan untuk:
1. Memperoleh data dari berbagai sumber dan populasi yang lebih beragam untuk meningkatkan generalisasi model.
2. Menguji algoritma machine learning lainnya seperti **Random Forest**, **Support Vector Machine (SVM)**, atau **Gradient Boosting** untuk membandingkan performa dan menemukan model yang lebih optimal.
3. Mengidentifikasi dan menciptakan fitur-fitur baru yang dapat meningkatkan kemampuan model dalam mendeteksi risiko penyakit jantung.
4. Mengimplementasikan model ke dalam sistem klinis nyata dan melakukan uji coba untuk menilai efektivitasnya dalam lingkungan operasional.

## Referensi

1. World Health Organization. (2021). **Cardiovascular Diseases (CVDs)**. Diakses dari [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

2. Kementerian Kesehatan Republik Indonesia. (2023). **Profil Kesehatan Indonesia 2023**. Jakarta: Kemenkes RI.

3. Nugraha, A., 2020. **Prediksi Penyakit Jantung Cardiovascular Menggunakan Model Algoritma Klasifikasi**. Repository Universitas Bina Sarana Informatika.

4. Hariyono, Putra Pratama. (2021). **Penerapan Machine Learning Untuk Prediksi Penyakit Jantung Menggunakan Metode Naïve Bayes Dan Decision Tree**. Universitas Airlangga Repository.

5. Bukhari, F., et al. (2023). **Deteksi Penyakit Jantung Menggunakan Metode Klasifikasi Decision Tree dan Regresi Logistik**. Jurnal SAKTI, 5(1), 41-49. 

6. Alhamad, A., et al. (2019). **Prediksi Penyakit Jantung Menggunakan Metode-Metode Machine Learning Berbasis Ensemble - Weighted Vote**. Jurnal Edukasi dan Penelitian Informatika, 5(3), 233-240. 

7. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python**. Journal of Machine Learning Research, 12, 2825-2830.

8. Zhang, Y., & Wallace, B. C. (2017). **A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification**. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 253-263).
