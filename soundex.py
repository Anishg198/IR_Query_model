# soundex.py
def soundex(word: str) -> str:
    """
    classic soundex implementation -> 4-char code
    """
    if not word:
        return ""
    w = word.upper()
    mappings = {
        "BFPV": "1", "CGJKQSXZ": "2",
        "DT": "3", "L": "4",
        "MN": "5", "R": "6"
    }
    def map_char(c):
        for k, v in mappings.items():
            if c in k:
                return v
        return ""  # vowels and ignored characters

    first = w[0]
    encoded = []
    for ch in w[1:]:
        code = map_char(ch)
        encoded.append(code)

    cleaned = []
    prev = None
    for code in encoded:
        if code == prev or code == "":
            prev = code
            continue
        cleaned.append(code)
        prev = code

    s = first + "".join(cleaned)
    s = (s[:4]).ljust(4, "0")
    return s

if __name__ == "__main__":
    print(soundex("robert"))   # expectation: r163
    print(soundex("rupert"))
    print(soundex("rubin"))
