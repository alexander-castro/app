from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from random import randrange

app = Flask(__name__)
CORS(app)

@app.route('/api/getmsg', methods=['GET'])
def respond():
    name = request.args.get("name", None)
    print(f"got name {name}")
    response = {}
    if not name:
        response["ERROR"] = "no name found, please send a name."
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"
    return jsonify(response)

@app.route('/api/patients', methods=['POST'])
def postPatients():
    param = request.data
    print(param)
    return jsonify({"Result": randrange(2)})

if __name__ == '__main__':
    app.run(threaded=True, port=5000)