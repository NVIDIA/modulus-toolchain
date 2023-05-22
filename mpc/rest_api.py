from bottle import run, request, post, get
import json


@get("/info")
def index():
    return {"a": 3}


@post("/infer")
def index():
    postdata = json.loads(request.body.read())
    print(postdata)  # this goes to log file only, not to client
    return "Hi {name}".format(name=postdata["name"])


run(host="localhost", port=8081, debug=True)
