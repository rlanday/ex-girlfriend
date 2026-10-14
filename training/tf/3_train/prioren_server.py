#!/usr/bin/env python3
"""Priors server that fills the line.

    python3 prioren_server.py --datei <path> [--port 8749] [--token <hex>]
                              [--erlaubt <ip>[,<ip>...]] [--minuten 180]
    (--datei = file, --erlaubt = allowed IPs, --minuten = minutes)

Why a server of its own: the data hosts have neither nginx nor aria2c nor
bbcp, and `python3 -m http.server` cannot serve range requests, without which
there are no parallel streams.  This server can; it is threaded and sends with
`os.sendfile` (kernel to socket, no copying through Python).  No apt, no
installation, nothing that stays on the machine.

Safeguards (the server carries 235 GB and is open to the network):
  * path token: the file lies under /<token>/prioren.bin; by default the
    token is 32 random hex characters, without it the answer is 404, and the
    token appears only in the output of this call.
  * IP allow list: --erlaubt switches to a positive list.
  * self shutdown: after --minuten the server stops by itself.
  * only GET and HEAD, only this one file, no directory, no writing.
"""
import argparse, os, secrets, socket, socketserver, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENDE = None
DATEI = None
TOKEN = None
ERLAUBT = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "neif-prioren/1"

    def log_message(self, fmt, *a):  # one line per request, to stderr
        sys.stderr.write("%s %s %s\n" % (time.strftime("%H:%M:%S"),
                                         self.client_address[0], fmt % a))

    def _erlaubt(self):
        if ERLAUBT and self.client_address[0] not in ERLAUBT:
            self.send_error(403, "not allowed")
            return False
        if self.path.split("?")[0].strip("/").split("/")[0] != TOKEN:
            self.send_error(404, "no such file")
            return False
        return True

    def do_HEAD(self):
        if not self._erlaubt():
            return
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(DATEI)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        if not self._erlaubt():
            return
        gesamt = os.path.getsize(DATEI)
        a, b = 0, gesamt - 1
        teil = False
        r = self.headers.get("Range")
        if r and r.startswith("bytes="):
            teil = True
            s, _, e = r[6:].partition("-")
            a = int(s) if s else 0
            b = int(e) if e else gesamt - 1
            if a > b or b >= gesamt:
                self.send_error(416, "bad range")
                return
        n = b - a + 1
        self.send_response(206 if teil else 200)
        self.send_header("Content-Length", str(n))
        self.send_header("Accept-Ranges", "bytes")
        if teil:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, gesamt))
        self.end_headers()
        fd = os.open(DATEI, os.O_RDONLY)
        try:
            self.wfile.flush()
            sock = self.connection
            rest, off = n, a
            while rest > 0:
                geschickt = os.sendfile(sock.fileno(), fd, off, min(rest, 1 << 24))
                if geschickt == 0:
                    break
                off += geschickt
                rest -= geschickt
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            os.close(fd)


def main():
    global DATEI, TOKEN, ERLAUBT
    p = argparse.ArgumentParser()
    p.add_argument("--datei", required=True)
    p.add_argument("--port", type=int, default=8749)
    p.add_argument("--token", default=None)
    p.add_argument("--erlaubt", default=None, help="comma-separated list of allowed IPs")
    p.add_argument("--minuten", type=int, default=180)
    a = p.parse_args()
    DATEI = a.datei
    if not os.path.isfile(DATEI):
        sys.exit("file missing: %s" % DATEI)
    TOKEN = a.token or secrets.token_hex(16)
    ERLAUBT = set(x.strip() for x in a.erlaubt.split(",")) if a.erlaubt else None

    socketserver.TCPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    srv.daemon_threads = True
    ip = socket.gethostbyname(socket.gethostname())
    print("### priors server on port %d, file %s (%d B)"
          % (a.port, DATEI, os.path.getsize(DATEI)), flush=True)
    print("URL   http://<public-ip>:%d/%s/prioren.bin" % (a.port, TOKEN), flush=True)
    print("local http://%s:%d/%s/prioren.bin" % (ip, a.port, TOKEN), flush=True)
    print("allowed: %s | self shutdown in %d min"
          % (",".join(sorted(ERLAUBT)) if ERLAUBT else "ALL IPs", a.minuten), flush=True)
    threading.Timer(a.minuten * 60, lambda: (print("### self shutdown", flush=True),
                                             srv.shutdown())).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    print("### server stopped", flush=True)


if __name__ == "__main__":
    main()
