from app.distilbert_service import predict_news

test_news = """
NASA has announced that the Moon will disappear from Earth's sky
for seven days next month due to a secret gravitational experiment.
"""

result = predict_news(test_news)

print("\nDistilBERT Prediction")
print("----------------------")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("Fake Probability:", result["fake_probability"])
print("Real Probability:", result["real_probability"])