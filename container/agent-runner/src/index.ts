/**
 * NanoClaw Agent Runner (OpenCode edition)
 * Runs inside a container, receives config via stdin, outputs result to stdout.
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent turn).
 *   Final marker after loop ends signals completion.
 *
 * Agent backend: opencode run CLI subprocess (opencode-go/glm-5)
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import readline from 'readline';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

// ---------------------------------------------------------------------------
// IPC helpers
// ---------------------------------------------------------------------------

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) { resolve(null); return; }
      const messages = drainIpcInput();
      if (messages.length > 0) { resolve(messages.join('\n')); return; }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

// ---------------------------------------------------------------------------
// Conversation archiving (preserved from Claude SDK runner)
// ---------------------------------------------------------------------------

interface ParsedMessage { role: 'user' | 'assistant'; content: string; }

function sanitizeFilename(summary: string): string {
  return summary.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 50);
}

function generateFallbackName(): string {
  const t = new Date();
  return `conversation-${t.getHours().toString().padStart(2, '0')}${t.getMinutes().toString().padStart(2, '0')}`;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null, assistantName?: string): string {
  const now = new Date();
  const formatDateTime = (d: Date) => d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : (assistantName || 'Assistant');
    const content = msg.content.length > 2000 ? msg.content.slice(0, 2000) + '...' : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Stdin helper
// ---------------------------------------------------------------------------

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// OpenCode event types (--format json nd-JSON stream)
// ---------------------------------------------------------------------------

interface OpenCodeEvent {
  type: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// runQuery: spawn `opencode run` and parse its nd-JSON event stream
// ---------------------------------------------------------------------------

async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  containerInput: ContainerInput,
): Promise<{ newSessionId?: string; closedDuringQuery: boolean; bufferedMessages: string[] }> {

  // Build inline config JSON injected via OPENCODE_CONFIG_CONTENT
  const mcpServerPath = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    'ipc-mcp-stdio.js',
  );

  const inlineConfig = JSON.stringify({
    model: process.env.OPENCODE_MODEL || 'opencode-go/glm-5',
    autoupdate: false,
    share: 'disabled',
    snapshot: false,
    provider: {
      'opencode-go': {
        options: {
          apiKey: process.env.OPENCODE_GO_API_KEY || '',
        },
      },
    },
    mcp: {
      nanoclaw: {
        type: 'local',
        command: ['node', mcpServerPath],
        environment: {
          NANOCLAW_CHAT_JID: containerInput.chatJid,
          NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
          NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        },
      },
    },
  });

  const args: string[] = ['run', prompt, '--format', 'json'];
  if (sessionId) {
    args.push('--session', sessionId, '--continue');
  }

  log(`Spawning: opencode ${args.slice(0, 3).join(' ')} ... (session=${sessionId || 'new'})`);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    OPENCODE_CONFIG_CONTENT: inlineConfig,
  };

  const proc = spawn('opencode', args, {
    cwd: '/workspace/group',
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let newSessionId: string | undefined;
  let assistantText = '';
  let closedDuringQuery = false;
  let ipcPolling = true;
  const bufferedIpcMessages: string[] = [];

  // Poll IPC for follow-up messages and _close sentinel during the query.
  // Messages are buffered in memory — NOT re-written to disk — to avoid the
  // re-queue loop where queued-*.json files get picked up and re-queued endlessly.
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query');
      closedDuringQuery = true;
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    if (messages.length > 0) {
      log(`IPC message(s) arrived during active query; buffering ${messages.length} for next turn`);
      bufferedIpcMessages.push(...messages);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  // Parse nd-JSON from stdout
  const rl = readline.createInterface({ input: proc.stdout!, crlfDelay: Infinity });

  const parsePromise = new Promise<void>((resolve) => {
    rl.on('line', (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;

      let event: OpenCodeEvent;
      try {
        event = JSON.parse(trimmed);
      } catch {
        // Non-JSON line — log and continue
        log(`[opencode stdout] ${trimmed}`);
        return;
      }

      log(`[event] type=${event.type} raw=${trimmed.slice(0, 500)}`);

      // Capture session ID from top-level sessionID field (present on every event)
      const topLevelSessionId = (event as { sessionID?: string }).sessionID;
      if (topLevelSessionId && !newSessionId) {
        newSessionId = topLevelSessionId;
        log(`Session ID: ${newSessionId}`);
      }

      // Session created events (legacy formats)
      if (event.type === 'session.created' || event.type === 'session') {
        const id = (event as { id?: string; session_id?: string }).id
          || (event as { session_id?: string }).session_id;
        if (id && !newSessionId) {
          newSessionId = id;
          log(`Session ID (legacy): ${newSessionId}`);
        }
      }

      // Assistant text parts (various event formats across OpenCode versions)
      if (event.type === 'message.part') {
        const part = event as { part?: { type?: string; text?: string } };
        if (part.part?.type === 'text' && part.part.text) {
          assistantText += part.part.text;
        }
      }

      // type=text — opencode run --format json emits text in event.part.text
      if (event.type === 'text') {
        const text = (event as { part?: { text?: string } }).part?.text;
        if (text) assistantText += text;
      }

      // text delta (alternative event format some versions emit)
      if (event.type === 'text.delta' || event.type === 'content.delta') {
        const delta = (event as { delta?: string; text?: string }).delta
          || (event as { text?: string }).text;
        if (delta) assistantText += delta;
      }

      // message.completed / step_finish — flush accumulated text as output
      if (event.type === 'message.completed' || event.type === 'assistant.message' || event.type === 'step_finish') {
        const text = assistantText.trim();
        assistantText = '';
        log(`Assistant message complete (${text.length} chars)`);
        if (text) {
          writeOutput({ status: 'success', result: text, newSessionId });
        }
      }
    });

    rl.on('close', resolve);
  });

  // Log stderr
  proc.stderr?.on('data', (chunk: Buffer) => {
    const lines = chunk.toString().trim().split('\n');
    for (const line of lines) {
      if (line) log(`[opencode stderr] ${line}`);
    }
  });

  const exitCodePromise = new Promise<number>((resolve) => {
    proc.on('close', (code) => resolve(code ?? 0));
    proc.on('error', (err) => {
      log(`opencode spawn error: ${err.message}`);
      resolve(1);
    });
  });

  await parsePromise;
  const exitCode = await exitCodePromise;
  ipcPolling = false;

  log(`opencode exited with code ${exitCode}`);

  if (exitCode !== 0 && !assistantText && !newSessionId) {
    throw new Error(`opencode process exited with code ${exitCode}`);
  }

  // If there was remaining text not flushed by message.completed event
  if (assistantText.trim()) {
    writeOutput({ status: 'success', result: assistantText.trim(), newSessionId });
    assistantText = '';
  }

  return { newSessionId, closedDuringQuery, bufferedMessages: bufferedIpcMessages };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  let sessionId = containerInput.sessionId;

  // Build initial prompt
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Query loop: run query → wait for IPC message → run new query → repeat
  try {
    while (true) {
      log(`Starting query (session: ${sessionId || 'new'})...`);

      const queryResult = await runQuery(prompt, sessionId, containerInput);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }

      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      // If messages arrived during the query, use them as the next prompt immediately
      if (queryResult.bufferedMessages.length > 0) {
        prompt = queryResult.bufferedMessages.join('\n');
        log(`Using ${queryResult.bufferedMessages.length} buffered message(s) as next prompt`);
        continue;
      }

      log('Query ended, waiting for next IPC message...');

      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage,
    });
    process.exit(1);
  }
}

main();
