import re

RE_URL = re.compile(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S")
RE_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

def clean(text, patterns=None):
    text = RE_URL.sub("", text)
    text = RE_IP.sub("", text)
    if patterns:
        for pattern in patterns:
            text = re.sub(pattern, "", text)
    return text.lower()

def balance_binary_data(texts, labels, seed=10, ratio=1):
    random.seed(seed)
    unique_labels = labels.unique()
    inds0 = list(texts[labels == unique_labels[0]].index)
    inds1 = list(texts[labels == unique_labels[1]].index)
    len_dict = {len(inds0): inds0,
                len(inds1): inds1}
    max_len = max(len_dict.keys())
    min_len = min(len_dict.keys())
    if (max_len / min_len) <= ratio:
        return None, None
    samples = random.sample(len_dict[max_len], k=int(ratio * min_len))
    samples.extend(len_dict[min_len])

    X = texts.loc[samples].reset_index(drop=True)
    y = labels.loc[samples].reset_index(drop=True)
    return X, y
