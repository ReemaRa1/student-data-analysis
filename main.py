import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ==============================
# 1) قراءة البيانات
# ==============================
df = pd.read_csv(r"C:\Users\reema\Downloads\student_data.csv")

print("📊 First 5 rows:")
print(df.head())

print("\n📊 Info:")
print(df.info())

print("\n📊 Description:")
print(df.describe())

# ==============================
# 2) إنشاء عمود المتوسط
# ==============================
df["Average"] = df[["G1", "G2", "G3"]].mean(axis=1)

# ==============================
# 3) أفضل وأسوأ طالب
# ==============================
top_student = df.loc[df["Average"].idxmax()]
low_student = df.loc[df["Average"].idxmin()]

print("\n🏆 Top Student:\n", top_student)
print("\n📉 Lowest Student:\n", low_student)

# ==============================
# 4) المتوسط العام
# ==============================
overall_avg = df["Average"].mean()
print("\n📌 Overall Average:", overall_avg)

# ==============================
# 5) العلاقة بين الغياب والدرجة
# ==============================
correlation = df["absences"].corr(df["G3"])
print("\n📊 Correlation (Absences vs Final Grade):", correlation)

# متوسط الدرجات حسب الجنس
gender_avg = df.groupby("sex")["G3"].mean()
print("\n📊 Average by Gender:\n", gender_avg)

# متوسط حسب وقت الدراسة
study_avg = df.groupby("studytime")["G3"].mean()
print("\n📊 Average by Study Time:\n", study_avg)

# ==============================
# 6) رسم بياني
# ==============================
df.plot(x="absences", y="G3", kind="scatter")
plt.title("Absences vs Final Grade")
plt.show()

# ==============================
# 7) Machine Learning 🔥
# الهدف: التنبؤ بالدرجة النهائية G3
# ==============================

# تحويل البيانات النصية إلى رقمية
df_ml = df.copy()
df_ml = pd.get_dummies(df_ml, drop_first=True)

# تحديد المدخلات (Features) والمخرجات (Target)
X = df_ml.drop("G3", axis=1)
y = df_ml["G3"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# بناء النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ
predictions = model.predict(X_test)

# تقييم النموذج
error = mean_absolute_error(y_test, predictions)
print("\n🤖 Model Error (MAE):", error)

# ==============================
# 8) تجربة التنبؤ
# ==============================
print("\n🎯 Sample Predictions:")
for i in range(5):
    print("Predicted:", round(predictions[i], 2), "| Actual:", y_test.iloc[i])