// smoke test: initialize + tools/list over stdio
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const serverPath = path.resolve(__dirname, "..", "dist", "index.js");

const child = spawn("node", [serverPath], { stdio: ["pipe", "pipe", "inherit"] });

const initReq = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "smoke-test", version: "0.0.1" },
  },
};
const listReq = { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} };

let buf = "";
const responses = [];

child.stdout.on("data", (chunk) => {
  buf += chunk.toString();
  let idx;
  while ((idx = buf.indexOf("\n")) !== -1) {
    const line = buf.slice(0, idx).trim();
    buf = buf.slice(idx + 1);
    if (!line) continue;
    responses.push(JSON.parse(line));
  }
});

child.stdin.write(JSON.stringify(initReq) + "\n");
child.stdin.write(
  JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }) + "\n"
);
child.stdin.write(JSON.stringify(listReq) + "\n");

setTimeout(() => {
  child.kill();
  const initRes = responses.find((r) => r.id === 1);
  const listRes = responses.find((r) => r.id === 2);

  if (!initRes?.result?.serverInfo) {
    console.error("FAIL: no valid initialize response", JSON.stringify(initRes));
    process.exit(1);
  }
  const tools = listRes?.result?.tools ?? [];
  const names = tools.map((t) => t.name).sort();
  const expected = ["get_trial", "search_trials", "trial_stats"];
  const ok = expected.every((n) => names.includes(n));

  console.log("initialize serverInfo:", JSON.stringify(initRes.result.serverInfo));
  console.log("tools/list names:", names);

  if (!ok) {
    console.error("FAIL: missing expected tools");
    process.exit(1);
  }
  console.log("SMOKE TEST PASSED");
  process.exit(0);
}, 1500);
