import pickle
with open('char_dict', 'rb') as f:
    char_dict = pickle.load(f)
    print(char_dict)
num_char = {}
with open('num_char', 'wb') as fr:
    for chars, nums in char_dict.items():
        num_char[nums] = chars
    print(num_char)
    pickle.dump(num_char, fr)
