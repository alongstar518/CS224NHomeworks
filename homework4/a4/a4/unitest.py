import utils

input = ["My name", "What is your", "How are you", "OK ?", "Fine"]
token = "XX"
input = list(map(lambda x: x.split(' '), input))
print(utils.pad_sents(input , token))