import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    MaxPooling2D,
    Dense,
    Dropout,
    Activation,
    Flatten,
    Convolution2D,
)
from tensorflow.keras.models import Sequential
import sqlite3

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)


def cleanData(doc):
    tokens = doc.split()
    table = str.maketrans("", "", punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = " ".join(tokens)
    return tokens


dataset = pd.read_csv("Dataset/dataset.csv", encoding="ISO-8859-1")
labels = dataset["Source"].unique().tolist()
symptoms = dataset.Target
diseases = dataset.Source

Y = [labels.index(disease) for disease in diseases]
X = [cleanData(symptom.strip().lower().replace("_", " ")) for symptom in symptoms]

# Vectorize symptoms
vectorizer = TfidfVectorizer(
    use_idf=True, smooth_idf=False, norm=None, decode_error="replace"
)
tfidf = vectorizer.fit_transform(X).toarray()
X = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names_out())
Y = np.asarray(Y)

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X.values[indices]
Y = Y[indices]
Y = to_categorical(Y)

# Reshape data for Convolutional Neural Network
X = X.reshape(X.shape[0], X.shape[1], 1, 1)

# Load model
classifier = Sequential()
classifier.add(
    Convolution2D(
        32, 1, 1, input_shape=(X.shape[1], X.shape[2], X.shape[3]), activation="relu"
    )
)
classifier.add(MaxPooling2D(pool_size=(1, 1)))
classifier.add(Convolution2D(32, 1, 1, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(1, 1)))
classifier.add(Flatten())
classifier.add(Dense(256, activation="relu"))
classifier.add(Dense(Y.shape[1], activation="softmax"))
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
classifier.fit(X, Y, batch_size=8, epochs=5, shuffle=True, verbose=1)

# Load drugs data from Excel

# drugs_data = pd.read_csv("Dataset/combined_drug_names.xlsx")
drugs_data = pd.read_csv("Dataset/disease_drugs.csv")

# Create a dictionary mapping diseases to drugs
disease_drugs = {}
for _, row in drugs_data.iterrows():
    disease = row["Disease"]
    drug = row["Drug"]
    if disease in disease_drugs:
        disease_drugs[disease].append(drug)
    else:
        disease_drugs[disease] = [drug]


def getDrugs(disease):
    return disease_drugs.get(disease, [])


def getDiet(filepath):
    diet = ""
    if os.path.exists("diets/" + filepath + ".txt"):
        with open("diets/" + filepath + ".txt", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                diet += lines[i] + "\n"
        file.close()
    else:
        with open("diets/others.txt", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                diet += lines[i] + "\n"
        file.close()
    return diet


@app.route("/")
def hello_world():
    return render_template("home.html")


@app.route("/logon")
def logon():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("signin.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/signup")
def signup():

    username = request.args.get("user", "")
    name = request.args.get("name", "")
    email = request.args.get("email", "")
    number = request.args.get("mobile", "")
    password = request.args.get("password", "")
    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",
        (username, email, password, number, name),
    )
    con.commit()
    con.close()
    return render_template("signin.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get("user", "")
    password1 = request.args.get("password", "")
    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "select `user`, `password` from info where `user` = ? AND `password` = ?",
        (
            mail1,
            password1,
        ),
    )
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")

    elif mail1 == "admin" and password1 == "admin":
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")


@app.route("/predict", methods=["GET"])
def predict():
    if request.method == "GET":
        question = request.args.get("mytext", False)
        question = question.strip("\n").strip()

        arr = question
        arr = arr.strip().lower()
        arr = arr.replace("_", " ")
        testData = vectorizer.transform([cleanData(arr)]).toarray()

        temp = testData.reshape(testData.shape[0], testData.shape[1], 1, 1)
        predict = classifier.predict(temp)
        predict = np.argmax(predict)
        output = labels[predict]

        diet = getDiet(output)

        drugs = getDrugs(output)

        print(question + " " + output)

        return jsonify(
            {
                "response": "Disease Predicted as "
                + output
                + "\n\n"
                + "Diet: "
                + diet
                + "\n\n"
                + "Drugs: "
                + ", ".join(drugs)
            }
        )


@app.route("/note")
def note():
    return render_template("notebook.html")


@app.route("/noteb")
def noteb():
    return render_template("notebook1.html")


if __name__ == "__main__":
    app.run(debug=False)
