
# Weather Prediction Project

Bu proje, geçmiş hava durumu verilerini kullanarak **yağış tahmini** yapan bir makine öğrenmesi sisteminin geliştirilmesini kapsamaktadır.  
Veriler üzerinde ön işleme, özellik mühendisliği, modelleme, model karşılaştırması ve hiperparametre optimizasyonu adımları gerçekleştirilmiştir.

---

## 1. Veri Kaynağı ve Birleştirme
- İki farklı dataset ile çalışıldı:
  - **Sıcaklık verileri** (`date`, `lat`, `long`, `elev`, `tmin`, `tmax`)
  - **Yağış verileri** (`date`, `lat`, `long`, `elev`, `precip`)
- Ortak sütunlar (`date`, `lat`, `long`) üzerinden birleştirme yapıldı.
- Eksik veriler incelendi, interpolasyon ile dolduruldu:
  ```python
  for col in ["tmin", "tmax"]:
      df[col] = (
          df.groupby(["lat", "long"], group_keys=False)[col]
            .apply(lambda s: s.interpolate(limit=3))
      )
  ```
- Ortalama sıcaklık (`tavg`) hesaplandı:
  ```python
  df["tavg"] = (df["tmin"] + df["tmax"]) / 2
  ```

---

## 2. Özellik Mühendisliği (Feature Engineering)
Modelin daha iyi öğrenebilmesi için yeni değişkenler eklendi:
- **Mevsimsellik**:
  ```python
  df["month"] = df["date"].dt.month
  df["doy"] = df["date"].dt.dayofyear
  df["sin_doy"] = np.sin(2*np.pi*df["doy"]/365.25)
  df["cos_doy"] = np.cos(2*np.pi*df["doy"]/365.25)
  ```
- **Gecikmeli (lag) ve hareketli toplam/ortalama özellikler**:
  ```python
  df["precip_lag1"] = df["precip"].shift(1)
  df["precip_lag3"] = df["precip"].shift(3)
  df["precip_lag7"] = df["precip"].shift(7)
  df["precip_sum3"] = df["precip"].rolling(3).sum()
  df["precip_sum7"] = df["precip"].rolling(7).sum()
  df["tavg_mean7"] = df["tavg"].rolling(7).mean()
  ```

---

## 3. Keşifsel Veri Analizi (EDA)
- **Korelasyon grafiği** ile sıcaklık, yağış ve ay değişkenleri arasındaki ilişkiler incelendi.
- **Scatter plot** ile tahmin ve gerçek değerler karşılaştırıldı.
- **Artık dağılımı (residual plot)** ile modelin hata dağılımı analiz edildi.

---

## 4. Modelleme
### Kullanılan Modeller:
- `LinearRegression`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `XGBRegressor` (XGBoost)
- `LGBMRegressor` (LightGBM) — yükleme sonrası denendi

### Eğitim Pipeline:
```python
pipe = Pipeline(steps=[
    ("prep", preprocessor),  # kategorik/numerik dönüşümler
    ("reg", model)           # tahmin modeli
])
```

### Karşılaştırma Sonuçları:
| Model           | RMSE   | R²     |
|-----------------|--------|--------|
| RandomForest    | 2.17   | 0.8955 |
| XGBoost         | 3.05   | 0.7940 |
| GradientBoost   | 4.08   | 0.6326 |
| LinearRegression| 4.57   | 0.5393 |

---

## 5. Hiperparametre Optimizasyonu
- En iyi sonucu veren **RandomForest** modeli için `RandomizedSearchCV` kullanıldı.
- Veri dengesini korumak için örnekleme yöntemi:
  ```python
  df["wet"] = (df["precip"] > 0).astype(int)
  key = df["month"].astype(str) + "_" + df["wet"].astype(str)
  sample_idx = df.groupby(key, group_keys=False)\
                 .apply(lambda g: g.sample(frac=0.2, random_state=42)).index
  df_sub = df.loc[sample_idx].copy()
  ```

---

## 6. Tahmin ve Değerlendirme
- Tahmin/gerçek değer grafiği çizildi.
- Artık dağılım grafiği ile model hataları incelendi.
---

