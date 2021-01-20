import pandas as pd

df1 = pd.read_csv("Emotions_merged.csv", engine = "python")
df2 = pd.read_csv("Emotions_new_with_sh.csv", engine = "python")

a_text = list(df1['Text'])
a_emo = list(df1['Emotion'])

b_text = list(df2['Text'])
b_emo = list(df2['Emotion'])

new_sentence = []
new_emotion = []

for i in range(len(a_text)):
    text = a_text[i]
    if text not in b_text:
        new_sentence.append(text)
        new_emotion.append(a_emo[i])

new_sentence = new_sentence + b_text
new_emotion = new_emotion + b_emo

adict = {"Text": new_sentence, "Emotion": new_emotion}
df = pd.DataFrame(adict)
df.to_csv("Emotions_merged_new.csv", index = False)
print(df.Emotion.value_counts())