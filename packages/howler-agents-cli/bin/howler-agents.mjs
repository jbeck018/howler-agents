#!/usr/bin/env node

/**
 * howler-agents CLI launcher
 *
 * Thin Node.js wrapper that ensures the Python howler-agents-core package
 * is installed, then proxies all commands to the Python CLI.
 *
 * Usage:
 *   npx howler-agents init                      # Set up skills, MCP, agents in any repo
 *   npx howler-agents serve                     # Start MCP server
 *   npx howler-agents install                   # Register with AI coding tools
 *   npx howler-agents evolve --domain coding    # Run evolution locally
 *   npx howler-agents install --host claude-code # Install for specific host
 *
 * The Python package is installed automatically via uv or pip on first use.
 */

import { execSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PYTHON_PACKAGE = "howler-agents-core[mcp]";
const VENV_DIR = join(homedir(), ".howler-agents", "venv");
const BIN_NAME = "howler-agents";

/**
 * Check if a command exists on PATH.
 */
function commandExists(cmd) {
  try {
    execSync(`command -v ${cmd}`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if howler-agents Python CLI is already available.
 *
 * Uses file-existence checks instead of spawning a Python process.
 * The old approach (`howler-agents --help`) imported the entire package
 * (litellm, numpy, sklearn) which caused multi-second startup and 100% CPU.
 */
function isInstalled() {
  // Check in our managed venv first (most common case after first install)
  const venvBin =
    process.platform === "win32"
      ? join(VENV_DIR, "Scripts", BIN_NAME)
      : join(VENV_DIR, "bin", BIN_NAME);
  if (existsSync(venvBin)) {
    return venvBin;
  }

  // Check if available on PATH via `command -v` (no Python import needed)
  try {
    const resolved = execSync(`command -v ${BIN_NAME}`, {
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 3_000,
    })
      .toString()
      .trim();
    if (resolved) {
      return BIN_NAME;
    }
  } catch {
    // not on PATH
  }

  return null;
}

/**
 * Install the Python package using the best available method.
 */
function installPython() {
  process.stderr.write(
    "howler-agents: Python package not found. Installing...\n"
  );

  // Method 1: uv (fastest, recommended)
  if (commandExists("uv")) {
    process.stderr.write("howler-agents: Installing via uv...\n");
    try {
      execSync(
        `uv venv "${VENV_DIR}" --python 3.12 2>/dev/null || uv venv "${VENV_DIR}"`,
        { stdio: "inherit" }
      );
      execSync(`uv pip install --python "${VENV_DIR}" "${PYTHON_PACKAGE}"`, {
        stdio: "inherit",
      });
      return join(
        VENV_DIR,
        process.platform === "win32" ? "Scripts" : "bin",
        BIN_NAME
      );
    } catch {
      process.stderr.write(
        "howler-agents: uv install failed, trying pip...\n"
      );
    }
  }

  // Method 2: pip with venv
  const python = commandExists("python3") ? "python3" : "python";
  if (commandExists(python)) {
    process.stderr.write(`howler-agents: Installing via ${python} + pip...\n`);
    try {
      execSync(`${python} -m venv "${VENV_DIR}"`, { stdio: "inherit" });
      const pip =
        process.platform === "win32"
          ? join(VENV_DIR, "Scripts", "pip")
          : join(VENV_DIR, "bin", "pip");
      execSync(`"${pip}" install "${PYTHON_PACKAGE}"`, { stdio: "inherit" });
      return join(
        VENV_DIR,
        process.platform === "win32" ? "Scripts" : "bin",
        BIN_NAME
      );
    } catch (err) {
      process.stderr.write(`howler-agents: pip install failed: ${err}\n`);
    }
  }

  // Method 3: pipx
  if (commandExists("pipx")) {
    process.stderr.write("howler-agents: Installing via pipx...\n");
    try {
      execSync(`pipx install "${PYTHON_PACKAGE}"`, { stdio: "inherit" });
      return BIN_NAME;
    } catch (err) {
      process.stderr.write(`howler-agents: pipx install failed: ${err}\n`);
    }
  }

  process.stderr.write(
    `\nhowler-agents: Could not install Python package automatically.\n` +
      `Please install manually:\n` +
      `  uv pip install "${PYTHON_PACKAGE}"    # recommended\n` +
      `  pip install "${PYTHON_PACKAGE}"        # alternative\n` +
      `  pipx install "${PYTHON_PACKAGE}"       # isolated\n\n`
  );
  process.exit(1);
}

// --- Main ---

const args = process.argv.slice(2);

// Special flag: --version shows the npm wrapper version
if (args.length === 1 && args[0] === "--npm-version") {
  const pkg = await import("../package.json", { with: { type: "json" } });
  process.stdout.write(`${pkg.default.version}\n`);
  process.exit(0);
}

// Find or install the Python CLI
let command = isInstalled();
if (!command) {
  command = installPython();
}

// Proxy all arguments to the Python CLI
const child = spawn(command, args, {
  stdio: "inherit",
  env: process.env,
});

child.on("error", (err) => {
  process.stderr.write(`howler-agents: Failed to start: ${err.message}\n`);
  process.exit(1);
});

child.on("close", (code) => {
  process.exit(code ?? 0);
});
