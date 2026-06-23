#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import net from "node:net";

const listenHost = process.env.LISTEN_HOST || "127.0.0.1";
const listenPort = Number(process.env.LISTEN_PORT || "11435");
const targetHost = process.env.TS_TARGET_HOST || "your-tailscale-host";
const targetPort = String(process.env.TS_TARGET_PORT || "11434");
const tailscaleBin =
  process.env.TAILSCALE_BIN ||
  "/opt/homebrew/bin/tailscale";

function getLocalApiCreds() {
  const out = execFileSync(tailscaleBin, ["debug", "local-creds"], {
    encoding: "utf8",
  }).trim();
  const match = out.match(/curl -u:([^ ]+) http:\/\/localhost:(\d+)\//);
  if (!match) {
    throw new Error(`Could not parse tailscale local-creds output: ${out}`);
  }
  return { token: match[1], port: Number(match[2]) };
}

let creds = getLocalApiCreds();

function dialViaTailscale(client) {
  client.pause();
  client.setNoDelay(true);

  const upstream = net.connect(creds.port, "127.0.0.1");
  upstream.setNoDelay(true);

  let headerBuffer = Buffer.alloc(0);
  let upgraded = false;

  upstream.on("connect", () => {
    const auth = Buffer.from(`:${creds.token}`).toString("base64");
    upstream.write(
      [
        "POST /localapi/v0/dial HTTP/1.1",
        `Host: localhost:${creds.port}`,
        "Connection: upgrade",
        "Upgrade: ts-dial",
        `Dial-Host: ${targetHost}`,
        `Dial-Port: ${targetPort}`,
        "Dial-Network: tcp",
        `Authorization: Basic ${auth}`,
        "Content-Length: 0",
        "",
        "",
      ].join("\r\n"),
    );
  });

  upstream.on("data", function onHandshake(chunk) {
    if (upgraded) return;
    headerBuffer = Buffer.concat([headerBuffer, chunk]);
    const headerEnd = headerBuffer.indexOf("\r\n\r\n");
    if (headerEnd === -1) return;

    const header = headerBuffer.subarray(0, headerEnd).toString("utf8");
    const leftover = headerBuffer.subarray(headerEnd + 4);
    const status = header.split("\r\n", 1)[0] || "";

    if (!status.includes("101")) {
      client.end(
        `HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nTailscale dial failed: ${status}\n${header}\n`,
      );
      upstream.destroy();
      return;
    }

    upgraded = true;
    upstream.off("data", onHandshake);
    if (leftover.length) client.write(leftover);
    client.resume();
    client.pipe(upstream);
    upstream.pipe(client);
  });

  const closeBoth = () => {
    client.destroy();
    upstream.destroy();
  };
  client.on("error", closeBoth);
  upstream.on("error", (err) => {
    if (!upgraded && /ECONNREFUSED|EPIPE/.test(err.code || "")) {
      try {
        creds = getLocalApiCreds();
      } catch {
        // Keep the original error path below.
      }
    }
    closeBoth();
  });
  client.on("close", () => upstream.destroy());
  upstream.on("close", () => client.destroy());
}

const server = net.createServer(dialViaTailscale);
server.listen(listenPort, listenHost, () => {
  console.log(
    `Forwarding http://${listenHost}:${listenPort} -> ${targetHost}:${targetPort} via Tailscale LocalAPI`,
  );
});
