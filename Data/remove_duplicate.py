import pandas as pd
import re

df = pd.read_csv(r"C:\Users\ishan\OneDrive\Desktop\Projects\Emotions-Detection\Data\Emotions_merged.csv", engine = "python")
text = list(df['Text'])
emo = list(df['Emotion'])

for i in range(len(text)):
    sentence = text[i]
    sentence = re.sub(r'[^\x00-\x7f]',r'', sentence)
    text[i] = sentence


adict = {"Text": text, 'Emotion': emo}
data = pd.DataFrame(adict)
data.to_csv("Emotions_merged_new.csv", index = False)