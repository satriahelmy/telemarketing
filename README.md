# Telemarketing Classification

## Blok Diagram Persiapan Data

## Blok Diagram Preprocessing

## Blok Diagram Pemodelan dan Evaluasinya

## Format message untuk melakukan prediksi via API
Kita perlu melakukan hit ke API `<host>:8080/predict`

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
