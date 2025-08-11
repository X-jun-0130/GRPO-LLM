#-*- coding: utf-8 -*-
import httpx
import json
client = httpx.Client()

reqUrl = "http://172.16.0.94:5053/v1/chat/completions"

headersList = {
 "Accept": "application/json",
 "Content-Type": "application/json"
}

def text_generate(messages, url=reqUrl):
    payload = {
        "model": "WiNGPT-Verifier",
        "max_tokens":2048,
        "messages":messages,
        "temperature": 0.0
        }
    data = client.post(url, json=payload, headers=headersList, timeout=240)
    try:
        reply = json.loads(data.text)["choices"][0]["message"]['content']
    except Exception as e:
        reply = ""
    return reply 
