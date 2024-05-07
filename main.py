from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index_template():
    return render_template("index.html")

#애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)