from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from arabert.preprocess import ArabertPreprocessor
app = Flask(__name__)

model_name = "AraT5v2TextCorrection"
preprocessor = ArabertPreprocessor(model_name="")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@app.route('/model/infere', methods=['POST'])
def infere():
    if request.method == 'POST':
        data = request.data.decode('utf-8')

        text = preprocessor.preprocess(data)

        result = pipeline(text,
                          pad_token_id=tokenizer.eos_token_id,
                          num_beams=3,
                          repetition_penalty=3.0,
                          max_length=256,
                          length_penalty=1.0,
                          no_repeat_ngram_size=3)[0]['generated_text']

        return result

if __name__ == '__main__':
    app.run(debug=True)
