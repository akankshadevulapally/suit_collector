#!/usr/env python3
import os
import logging
import http.server as server
from game import game 
import json
import numpy as np

env = game()

class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
    """
    SimpleHTTPServer with added bonus of:

    - handle POST requests
    - log headers in GET request if requried
    """
    def do_GET(self):
        server.SimpleHTTPRequestHandler.do_GET(self)
        #logging.warning(self.headers)

    def do_POST(self):
        """handle HTTP POST request"""
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body)
        output = env.process_request(data)
        print(output)
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(output, default=self.serialize_int32), "utf8"))
    
    def serialize_int32(self,obj):
        if isinstance(obj, np.int32):
            return int(obj)
        raise TypeError ("Type %s is not serializable" % type(obj))

if __name__ == '__main__':
    server.test(HandlerClass=HTTPRequestHandler)