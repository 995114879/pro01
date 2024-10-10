from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# 加载预训练的Tokenizer和Model
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

@app.route('/get_response', methods=['POST'])
def get_response():
    # 获取用户输入
    user_input = request.json['input']
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回响应
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
