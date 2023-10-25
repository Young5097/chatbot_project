from flask import Flask, request, render_template
import torch
import joblib
import os

tokenizer = joblib.load("C:/Users/Kim/Desktop/일상대화챗봇/tokenizer.pkl")
model = torch.load(
    "C:/Users/Kim/Desktop/일상대화챗봇/model2.pt", map_location=torch.device("cpu")
)

app = Flask(__name__)
app.debug = True
RESULT_FOLDER = os.path.join("static")
app.config["RESULT_FOLDER"] = RESULT_FOLDER


def gpt_test(q, tokenizer, model):
    Q_TKN = "<usr>"  # 사용자 질문 토큰
    A_TKN = "<sys>"
    SENT = "<unused1>"
    EOS = "</s>"
    with torch.no_grad():
        q = q.strip()
        a = ""
        while 1:
            input_ids = torch.LongTensor(
                tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)
            ).unsqueeze(dim=0)
            pred = model(input_ids)  # 모델에 입력을 주어 대답 생성
            pred = pred.logits.cpu()
            gen = tokenizer.convert_ids_to_tokens(
                torch.argmax(pred, dim=-1).squeeze().numpy().tolist()
            )[
                -1
            ]  # 모델 출력에서 다음 토큰을 선택해 대답 생성함
            if gen == EOS:  # 대답 토큰이 종료 토큰과 일치될 때(대답 생성이 종료될 때)
                break
            a += gen.replace("▁", " ")
        text = a.strip()
    return text


conversation_history = []


# Main page
@app.route("/", methods=["GET", "POST"])
def main():
    text = ""
    answer = ""
    if request.method == "POST":
        text = request.form.get("message")
        answer = gpt_test(text, tokenizer, model)
        conversation_history.append(("나", text))
        conversation_history.append(("챗봇", answer))
    return render_template(
        "main.html", text=text, answer=answer, conversation_history=conversation_history
    )


if __name__ == "__main__":
    app.debug = True
    app.run()
