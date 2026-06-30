import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

review = "This product is not bad"

prediction = model.predict([review])[0]

if prediction == 1:
    print("Positive")
else:
    print("Negative")