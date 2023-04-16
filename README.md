# Telemarketing Classification

## Latar Belakang

Selain memberikan manfaat sosial, Bank sebagai sebuah perusahaan memiliki kebutuhan untuk mendapatkan keuntungan dari setiap usahanya. Salah satu caranya adalah menyalurkan kredit dan investasi surat berharga. Namun, untuk menyalurkan kredit tentu saja membutuhkan sumber dana yang mencukupi. Sumber dana bank sendiri terdiri dari 3 jenis, yaitu modal sendiri, modal dari pihak lembaga lain, dan modal dari masyarakat.

Oleh sebab itu, dilakukan usaha-usaha yang bisa dilakukan untuk menarik minat masyarakat untuk menyetorkan dananya ke bank. Salah satunya adalah dengan melakukan campaign telemarketing. Namun, telemarketing-pun juga membutuhkan resource yang tidak sedikit baik itu biaya, tenaga dan waktu. Dibutuhkan targeting yang tepat terhadap customer yang akan melakukan campaign telemarketing. Sehingga resource yang dibutuhkan tidak sia-sia dan bank mampu mendapatkan modal dengan lebih banyak sehingga profit perusahaan juga akan meningkat.

## Objectives

Dari permasalahan tersebut, dibuatlah klasifikasi yang bertujuan untuk memprediksi apakah nasabah akan berlangganan deposit atau tidak yang nantinya akan dapat digunakan untuk mengoptimalisasi targeting pada campaign telemarketing sehingga resource yang ada dapat dimanfaatkan kepada target yang sesuai dan menjadikan peningkatan profit pada perusahaan.

## Dataset
Dataset yang digunakan adalah dataset bank telemarketing dengan feature sebagai berikut.

- `age` : umur klien (numeric)
- `job` : pekerjaan (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
- `marital` : status perkawinan (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
- `education` : pendidikan (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
- `default`: memiliki kredit yang default? (binary: "yes", "no")
- `housing`: memiliki pinjaman perumahan? (binary: "yes", "no")
- `loan`: memiliki pinjaman pribadi? (binary: "yes", "no")

<br>

**Kondisi komunikasi dengan campaign terakhir**
- `contact`: tipe kontak (categorical: "unknown", "telephone", "cellular")
- `day`: hari terakhir kontak (numeric)
- `month`: bulan terakhir kontak (categorical: "jan", "feb", "mar", ..., "nov", "dec")
- `duration`: durasi kontak terakhir dalam seconds (numeric)

<br>

**Atribut/Fitur lain**
- `campaign`: jumlah kontak dalam campaign(numeric, includes last contact)
- `pdays`: jumlah hari yang berlalu setelah klien terakhir dihubungi dari kampanye sebelumnya (numeric, -1 means client was not previously contacted)
- `previous`: jumlah kontak yang dilakukan sebelum kampanye ini(numeric)
- `poutcome`: hasil dari kampanye pemasaran sebelumnya (categorical: "unknown","other","failure","success")

<br>

**Kondisi sosial ekonomi**
- `emp.var.rate`: employment variation rate - quarterly indicator (numeric)
- `cons.price.idx`: consumer price index - monthly indicator (numeric)     
- `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)     
- `euribor3m`: euribor 3 month rate - daily indicator (numeric)
- `nr.employed`: number of employees - quarterly indicator (numeric)

<br>

**Output variable (desired target)**
- `y` - apakah klien berlangganan deposit? (binary: "yes","no")

## Overall Workflow
![](https://github.com/satriahelmy/telemarketing/blob/main/image/workflow.png)

## Blok Diagram Persiapan Data

![](https://github.com/satriahelmy/telemarketing/blob/main/image/data_pipeline.png)

## Blok Diagram Preprocessing dan Feature Engineering

![](https://github.com/satriahelmy/telemarketing/blob/main/image/preprocessing.png)

## Blok Diagram Pemodelan dan Evaluasinya

![](https://github.com/satriahelmy/telemarketing/blob/main/image/modelling.png)

## Format message untuk melakukan prediksi via API
Kita perlu melakukan hit ke API `<host>:8080/predict` dengan method POST

```
{
    "age" : 41,
    "duration" : "1575",
    "campaign" : 1,
    "pdays" : 999,
    "previous" : 0,
    "job" : "blue-collar",
    "marital" : "divorced" ,
    "education" : "basic.4y",
    "default" : "unknown",
    "housing" : "yes",
    "loan" : "no",
    "contact" : "telephone",
    "month" : "may",
    "day_of_week" : "mon",
    "poutcome" : "nonexistent",
    "empvarrate" : 1.1,
    "conspriceidx" : 93.994,
    "consconfidx" : -36.4,
    "euribor3m" : 4.857,
    "nremployed" : 5191
} 
```

## Format message return dari API
```
{
    "res": "yes",
    "error_msg": ""
}
```

## Tampilan Website (Streamlit)
Selain API, terdapat juga tampilan dalam bentuk website. Cara mengaksesnya adalah sebagai berikut : `<host>:8501`
![](https://github.com/satriahelmy/telemarketing/blob/main/image/website.jpg)

## Cara menjalankan machine learning di lokal Computer

### Retraining model

Terdapat beberapa tahapan dalam melakukan retraining model, antara lain `data_pipeline`, `preprocessing` dan `modelling`. Cara menjalankannya adalah sebagai berikut:

1. Data Pipeline

&nbsp;&nbsp;&nbsp; `python src/data_pipeline.py`

2. Preprocessing

&nbsp;&nbsp;&nbsp; `python src/preprocessing.py`

3. Modelling

&nbsp;&nbsp;&nbsp; `python src/modelling.py`

### Menjalankan API
Terdapat 2 cara menjalankan API, yaitu dengan menggunakan uvicorn atau menggunakan docker.

- Jika menggunakan uvicorn, maka jalankan script berikut

&nbsp;&nbsp;&nbsp; `python src/api.py`

- Jika menggunakan docker, maka jalankan script berikut (perlu instal docker terlebih dahulu)

&nbsp;&nbsp;&nbsp; `docker-compose up -d`

## Simpulan
- Klasifikasi bank telemarketing telah dapat digunakan (dideploy) dalam bentuk API dan Website (Streamlit)
- Klasifikasi bank telemarketing memiliki performa macro average F1 sebesar 0.78

## Reference

- Dataset : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- Citation :
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001
- Penjelasan medium : https://medium.com/@helmysmp/ml-process-klasifikasi-client-berlangganan-deposit-dengan-data-telemarketing-b1c511ced19c
