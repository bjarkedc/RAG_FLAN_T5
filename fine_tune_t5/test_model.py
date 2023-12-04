from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast
import torch
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def eval(gold, pred):
    """
    An answer is considered correct if at least half of the gold
    tokens are in the prediction. Note that this is a shortcut, 
    and will favor long answers.
    """
    gold = set(gold.strip().lower().replace('.', '').split(' '))
    pred = set(pred.strip().lower().replace('.', '').split(' '))
    return len(gold.intersection(pred)) >= len(gold)/2


# iterate over model folders
for folder in os.listdir('.'):
    if "flant5-base" in folder:
        lm = folder
        print(lm)

        lm = folder
        lang_model = T5ForConditionalGeneration.from_pretrained(lm)
        lang_model.to(DEVICE)
        tokenizer = T5TokenizerFast.from_pretrained(lm)


        questions = open('fine_tune_t5/questions.txt', encoding="latin1").readlines()
        answers = open('fine_tune_t5/answers.txt', encoding="latin1").readlines()
        # questions = ['What is the capital of Denmark ?', 'What is the square root of 4 ?', 'What is the average size of a human ?', 'What is the cause of the housing crisis in London ?', 'What is the price of 1L of milk at Netto ?', 'How many insect species are there ?', 'Does the pope shit in the woods ?']
        # answers = ['Copenhagen', '2', '1.7 meters', 'Investors and corrupt politicans', '10 DKK', '1 million', 'yes']

        prefixes = ['Answer this Star Wars trivia question:']
        postfixes = ['']
        # prefixes = ['']
        # postfixes = ['']

        for prefix, postfix in zip(prefixes, postfixes):
            correct = 0
            for question, answer in zip(questions, answers):
                question = prefix + ' ' + question.strip() + ' ' + postfix
                tokked = tokenizer(question.strip(), return_tensors='pt')['input_ids']
                tokked = tokked.to(DEVICE)
                generated_ids = lang_model.generate(tokked, max_new_tokens=20)
                tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # print()
                # print(question)
                # print(' '.join(tokens))
                # writer question and answer to file
                with open(f"{lm}_question_answer.txt", "a+") as f:
                    f.write(f"{question}\n")
                    f.write(f"{answer}\n")
                    f.write(f"{' '.join(tokens)}\n")
                correct += int(eval(answer, ' '.join(tokens)))

            print(str(correct) + ' out of ' + str(len(answers)) + ' correct')