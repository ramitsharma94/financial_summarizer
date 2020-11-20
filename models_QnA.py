
from transformers import BertTokenizerFast
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import pipeline

tokenizer_QnA = AutoTokenizer.from_pretrained("./Trained_Models/my_model3")
model_QnA = AutoModelForQuestionAnswering.from_pretrained("./Trained_Models/my_model3")

def answer(text, question):
    nlp_pipline = pipeline('question-answering', model=model_QnA, tokenizer=tokenizer_QnA)
    nlp_input = {'question': question, 'context': text}
    result = nlp_pipline(nlp_input)
    return result['answer']
