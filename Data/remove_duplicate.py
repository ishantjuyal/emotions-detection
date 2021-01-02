import pandas as pd

df = pd.read_csv(r"C:\Users\ishan\OneDrive\Desktop\Projects\Emotions-Detection\Data\Emotions_merged.csv", engine = "python")
text = df['Text']
emo = df['Emotion']

text_new = []
emo_new = []

for i in range(len(text)):
    if text[i] not in text_new:
        text_new.append(text[i])
        emo_new.append(emo[i])

print(len(text))
print(len(text_new))

adict = {"Text": text_new, 'Emotion': emo_new}
data = pd.DataFrame(adict)
data.to_csv("Emotions_merged_new.csv", index = False)