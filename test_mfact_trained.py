from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F


# Load a pretrained NLI model (like BERT fine-tuned on MNLI dataset)
model_name = """/home/avramit/.cache/huggingface/hub/models--dicta-il--alephbertgimmel-base/snapshots/973a5f8bb492c0525c21b4f9dd3d18f9ddc02dad"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
doc = "הצעיר בן ה-21 נמצא בטננדיייס מאז קיץ 2013, אך ערך רק הופעה אחת בהרכב הראשון כמחליף. האנגלי אויונוגה החל את הקריירה שלו עם טוטנהאם והושאל למועדונים מייפה בפינלנד, בורי וסנט ג'ונסטון לפני ששוחרר. הוא בילה עונה עם מועדון הקונפרנס הייז ויאדינג יונייטד לפני שעבר לטננדיייס. אויונוגה בילה את החצי השני של העונה שעברה בהשאלה במועדון הקונפרנס סאות' בורהאם ווד. כעת הוא הצטרף לקבוצתו של ג'ימי ניקול, בלו ברזיל, בליגת האליפות הסקוטית עד ינואר ויכול לערוך את הבכורה שלו במשחק גביע הליגה ביום שלישי נגד פאלקירק."
summary = "החלוץ לשעבר של טוטנהאם הוטספור, קודוס אויונוגה, הצטרף לקאוודנביט בהשאלה מדנדי יונייטד."
inputs = tokenizer(doc, summary, return_tensors="pt")
logits = model(**inputs).logits
prediction = torch.argmax(logits, dim=-1)
probs = F.softmax(logits, dim=-1)
print("Probabilities:", probs)
print(logits)
print(prediction)