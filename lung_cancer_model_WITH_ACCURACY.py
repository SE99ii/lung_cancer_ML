import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('lung_cancer.csv')
df.describe().to_csv("lung_cancer_statistics.csv")

sns.countplot(data=df, x="LUNG_CANCER")
plt.title("Distribution of Lung Cancer Cases")
plt.xlabel("LUNG_CANCER")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("lung_cancer_distribution.png")
plt.show()

le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])       # M=1, F=0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("data/preprocessed_data/X.csv", index=False)
X_test.to_csv("data/preprocessed_data/X_test.csv", index=False)
y_train.to_csv("data/preprocessed_data/Y.csv", index=False)
y_test.to_csv("data/preprocessed_data/Y_test.csv", index=False)

models = {
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Naive_Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Linear_Regression": LinearRegression()
}

print("Model Accuracy Scores:")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if name == "Linear_Regression":
        y_pred = [round(val) for val in y_pred]

    pd.DataFrame(y_pred, columns=["prediction"]).to_csv(f"data/Results/predictions_{name}_model.csv", index=False)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Metrics:")
    print(f"  Accuracy : {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall   : {recall:.2f}")
    print(f"  F1-Score : {f1:.2f}")
