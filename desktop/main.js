const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const PORT = 5858;
const PROJECT_ROOT = path.join(__dirname, "..");
const isWin = process.platform === "win32";
const pythonPath = path.join(
  PROJECT_ROOT,
  "venv",
  isWin ? "Scripts/python.exe" : "bin/python"
);

let pythonProcess = null;
let mainWindow = null;

function waitForServer(port, timeoutMs = 120000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const check = () => {
      const request = http.get(`http://127.0.0.1:${port}/`, (response) => {
        response.resume();
        resolve();
      });

      request.on("error", () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error("Flask server did not start in time."));
          return;
        }
        setTimeout(check, 500);
      });
    };

    check();
  });
}

function startFlaskServer() {
  pythonProcess = spawn(pythonPath, ["app.py", "--port", String(PORT)], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[flask] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[flask] ${data.toString().trim()}`);
  });

  pythonProcess.on("exit", (code) => {
    console.log(`[flask] exited with code ${code}`);
    pythonProcess = null;
  });
}

function stopFlaskServer() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

async function createWindow() {
  startFlaskServer();
  await waitForServer(PORT);

  mainWindow = new BrowserWindow({
    width: 1120,
    height: 820,
    minWidth: 860,
    minHeight: 640,
    title: "Stuffy Identifier",
    backgroundColor: "#FFF5F7",
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  await mainWindow.loadURL(`http://127.0.0.1:${PORT}/`);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  stopFlaskServer();
  app.quit();
});

app.on("before-quit", () => {
  stopFlaskServer();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
