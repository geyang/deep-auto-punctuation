import os


def get_content(fn):
    with open(fn, 'r') as f:
        source = ""
        for line in f:
            source += line
    return source


def source_gen(path="./engadget_data/", start=None, end=None):
    child, folders, files = list(os.walk(path))[0]
    for fn in sorted(files, key=lambda fn: os.path.getsize(path + fn)):
        if fn[0] is ".":
            pass
        else:
            src = get_content(path + fn)
            yield fn, src


def train_gen():
    yield from [_ for i, _ in enumerate(source_gen()) if (716 - i) % 2000 not in [1998, 1995]]


def validation_gen():
    yield from list(source_gen())[-2:0:-2000][::-1]


def test_gen():
    yield from list(source_gen())[-5:0:-2000][::-1]


if __name__ == "__main__" and False:
    fig = plt.figure(figsize=(12, 2))
    plt.subplot(131)
    plt.plot([len(src) for fn, src in source_gen()], linewidth=3, alpha=.7)
    plt.title("Distribution of Document Length")
    plt.xlabel('Document Index')
    plt.ylabel('Document Length')
    plt.subplot(132)
    plt.plot([len(src) for fn, src in validation_gen()], linewidth=3, alpha=.7)
    plt.title("Validation Set")
    plt.xlabel('Document Index')
    plt.ylabel('Document Length')
    plt.subplot(133)
    plt.plot([len(src) for fn, src in test_gen()], linewidth=3, alpha=.7)
    plt.title("Test Set")
    plt.xlabel('Document Index')
    plt.ylabel('Document Length')

    plt.tight_layout()
    plt.show()


def apply_punc(text_input, punctuation):
    assert len(text_input) == len(punctuation), "input string has differnt length from punctuation list" + "".join(
        text_input) + str(punctuation) + str(len(text_input)) + ";" + str(len(punctuation))
    result = ""
    for char1, char2 in zip(text_input, punctuation):
        if char2 == "<cap>":
            result += char1.upper()
        elif char2 == "<nop>":
            result += char1
        else:
            result += char2 + char1
    return result


if __name__ == "__main__":
    result = apply_punc("t s", ['<cap>', '<nop>', ','])
    print(result)
    assert result == "T ,s", "apply_func result incorrect"


def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    input_length = len(string_input)
    i = 0
    while i < input_length:
        char = string_input[i]
        if char.isupper():
            output_source.append("<cap>")
            input_source.append(char.lower())

        if char in output_chars:
            output_source.append(char)
            if i < input_length - 1:
                input_source.append(string_input[i + 1])
            else:
                input_source.append(" ")
            i += 1

        if not char.isupper() and char not in output_chars and char in input_chars:
            input_source.append(char)
            output_source.append("<nop>")

        i += 1
    return input_source, output_source


if __name__ == "__main__" and False:
    i, o = extract_punc("ATI'd. I'm not sure if $10 is enough. ", input_chars, output_chars)
    print(i)
    print(o)
    result = apply_punc("".join(i), o)
    print(result)

import math


def fuzzy_chunk_len(max_len, seg_len):
    return max_len // max(math.ceil(max_len / seg_len) - 1, 1)


def chunk_gen(seq_length, src_list, filler=[" "]):
    s_l = len(src_list)
    b_n = math.ceil(s_l / seq_length)
    s_pad = src_list + filler * (b_n * seq_length - s_l)
    for i in range(b_n):
        yield s_pad[i * seq_length: (i + 1) * seq_length]


def batch_gen(src_gen, bsize):
    batch = []
    for i, (fn, src) in enumerate(src_gen):
        batch.append(src)
        max_len = len(src)
        if i % bsize == bsize - 1:
            yield max_len, batch
            batch = []
