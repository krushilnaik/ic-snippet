from flask import Flask, jsonify, request

from graph import graph

app = Flask(__name__)


@app.post("/test")
def test():
    payload = request.get_json()

    final = graph.invoke({"request": payload["text"]})
    # final["messages"] = ""
    del final["messages"]

    return jsonify(final)
