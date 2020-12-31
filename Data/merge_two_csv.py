import pandas as pd

def merge(url1, url2):
    df1 = pd.read_csv(url1, engine = 'python')
    df2 = pd.read_csv(url2, engine = 'python')

    frames = [df1, df2]
    result = pd.concat(frames)

    text = list(result['sentence'])

    unique_text = list(set(text))

    print("Original", len(text))
    print("Unique", len(unique_text))

    adict = {'sentence': unique_text}
    data = pd.DataFrame(adict)
    data.to_csv("merged_label.csv", index = False)
    
url1 = 'labelled_50.csv'
url2 = 'labelled_trial.csv'

merge(url1, url2)

