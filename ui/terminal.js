/* Shell page. The websocket stays on the manager, which attaches the shell.
   This page never receives the Docker socket. */

const params = new URLSearchParams(location.search);
const name = params.get("name") || "";
const title = document.querySelector("#title");
const status = document.querySelector("#status");
const host = document.querySelector("#term");
const safeName = /^[a-z0-9][a-z0-9_-]{0,62}$/.test(name);

title.textContent = safeName ? name : "Shell";

if (!safeName) {
  status.textContent = "Missing instance name.";
} else {
  const term = new Terminal({
    cursorBlink: true,
    fontSize: 15,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    theme: { background: "#000000", foreground: "#f5f5f7", cursor: "#e7c27a" },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(host);
  fit.fit();

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${proto}//${location.host}/api/instances/${encodeURIComponent(name)}/terminal`
  );
  socket.binaryType = "arraybuffer";

  function sendResize() {
    if (socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
  }

  socket.addEventListener("open", () => {
    status.textContent = "Connected";
    fit.fit();
    sendResize();
    term.focus();
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") term.write(event.data);
    else term.write(new Uint8Array(event.data));
  });
  socket.addEventListener("close", () => {
    status.textContent = "Disconnected";
    term.write("\r\n[session closed]\r\n");
  });
  socket.addEventListener("error", () => {
    status.textContent = "Connection failed. Start the instance and sign in again.";
  });
  term.onData((data) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(data);
  });
  term.onResize(sendResize);
  window.addEventListener("resize", () => {
    fit.fit();
    sendResize();
  });
}
