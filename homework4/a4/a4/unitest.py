import utils
'''
input = ["My name", "What is your", "How are you", "OK ?", "Fine"]
token = "XX"
input = list(map(lambda x: x.split(' '), input))
print(utils.pad_sents(input , token))
'''



from nltk.translate.bleu_score import sentence_bleu
reference = ["Love can always find a way".split(' '),"Love makes anything possible".split(' ')]
candidate = "The love can always do".split(' ')
score = sentence_bleu(reference, candidate, weights=(0.5,0.5))
print(score)

candidate = "Love can make anything possible".split(' ')
score = sentence_bleu(reference, candidate, weights=(0.5,0.5))
print(score)