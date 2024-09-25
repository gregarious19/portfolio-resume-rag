from django.shortcuts import render

from django.http import HttpResponse
from . import RAG
import json


def index(request):
    if request.method == "GET":
        try:
            data = request.GET.get("query")
            # console.log(data)
        except json.JSONDecodeError:
            # Handle invalid JSON data
            pass
        print(data)
        res = {"response": "I am Pranay"}

        # response = HttpResponse(RAG.pipeline(data), content_type="text/plain")
        response = HttpResponse(json.dumps(res), content_type="application/json")
        response["Access-Control-Allow-Origin"] = "*"
        return response


def txt(request):
    res = {"response": "I am Pranay"}
    response = HttpResponse(json.dumps(res), content_type="application/json")
    return response
