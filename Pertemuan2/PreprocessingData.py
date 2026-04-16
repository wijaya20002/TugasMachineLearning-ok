from PrediksiHargaRumah import X_train, X_test
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),
         ['luas_tanah', 'jarak_ke_pusat_kota', 'tahun_dibangun'])
    ]
)

# preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing berhasil")