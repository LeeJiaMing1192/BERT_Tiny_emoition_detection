import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny")

# Load config
config = BertConfig.from_pretrained("C:\\Users\\ADMIN\\Desktop\\custom_vtuber\\bert-tiny")
config.num_labels = 6  # Set this to your number of emotion classes

# Load model
model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=torch.load("./best_final_emotion_model.bin", map_location=DEVICE)
)
model = model.to(DEVICE)
model.eval()

# Define label mapping (reverse from training)
id2label = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'fear',
    5: 'surpise'
}  # Make sure this matches your original label set

# Inference function
def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    return id2label[prediction]

# Example usage
text_input = "Oh no......"
predicted_emotion = predict_emotion(text_input)
print(f"Predicted Emotion: {predicted_emotion}")
