"""Self-contained Live Systolic Matrix HUD HTML renderer for TPUForge Serve."""

from __future__ import annotations

import json
from typing import Any

_HUD_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>TPUForge Serve: Live Systolic Matrix HUD</title>
  <script>
    (function () {
      try {
        var params = new URLSearchParams(window.location.search);
        var urlTheme = params.get("theme");
        var stored = window.localStorage ? window.localStorage.getItem("tpuforge-theme") : null;
        var chosen = urlTheme === "dark" || urlTheme === "light" ? urlTheme : stored;
        var meta = document.querySelector('meta[name="color-scheme"]');
        if (chosen === "dark" || chosen === "light") {
          if (meta) {
            meta.content = chosen;
          }
          document.documentElement.setAttribute("data-theme", chosen);
        }
      } catch (_err) {
        /* Ignore storage errors inside restricted iframes. */
      }
    })();
  </script>
  <style>
    @layer reset, base, theme, components, utilities;

    @layer reset {
      html, body, main, header, aside, footer, section, article,
      div, form, fieldset, legend, label, input, textarea, select, button,
      table, thead, tbody, tr, th, td, dialog, pre, code, h1, h2, h3, p, ul, li {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        padding: 0;
      }
    }

    @layer theme {
      :root {
        color-scheme: light dark;
        --bg: #f4f6f9;
        --surface: #ffffff;
        --surface-2: #edf1f7;
        --border: #d0d7de;
        --text: #111827;
        --text-muted: #4b5563;
        --accent-blue: #1a73e8;
        --accent-green: #188038;
        --accent-red: #c5221f;
        --accent-amber: #b06000;
        --cell-idle: #e2e8f0;
        --cell-root: #1a73e8;
        --cell-accepted: #188038;
        --cell-rejected: #c5221f;
        --cell-text: #ffffff;
        --code-bg: #f1f5f9;
        --focus-ring: #1a73e8;
        --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Liberation Sans", sans-serif;
        --font-mono: "JetBrains Mono", "Cascadia Code", "Fira Code", ui-monospace, SFMono-Regular, monospace;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #0b0f17;
          --surface: #121824;
          --surface-2: #1a2232;
          --border: #2a3447;
          --text: #e6edf3;
          --text-muted: #9aa7b8;
          --accent-blue: #4285f4;
          --accent-green: #34a853;
          --accent-red: #ea4335;
          --accent-amber: #fbbc04;
          --cell-idle: #242d3d;
          --cell-root: #4285f4;
          --cell-accepted: #34a853;
          --cell-rejected: #ea4335;
          --cell-text: #0b0f17;
          --code-bg: #0d131d;
          --focus-ring: #8ab4f8;
        }
      }

      @supports (color: light-dark(white, black)) {
        :root {
          --bg: light-dark(#f4f6f9, #0b0f17);
          --surface: light-dark(#ffffff, #121824);
          --surface-2: light-dark(#edf1f7, #1a2232);
          --border: light-dark(#d0d7de, #2a3447);
          --text: light-dark(#111827, #e6edf3);
          --text-muted: light-dark(#4b5563, #9aa7b8);
          --accent-blue: light-dark(#1a73e8, #4285f4);
          --accent-green: light-dark(#188038, #34a853);
          --accent-red: light-dark(#c5221f, #ea4335);
          --accent-amber: light-dark(#b06000, #fbbc04);
          --cell-idle: light-dark(#e2e8f0, #242d3d);
          --cell-root: light-dark(#1a73e8, #4285f4);
          --cell-accepted: light-dark(#188038, #34a853);
          --cell-rejected: light-dark(#c5221f, #ea4335);
          --cell-text: light-dark(#ffffff, #0b0f17);
          --code-bg: light-dark(#f1f5f9, #0d131d);
          --focus-ring: light-dark(#1a73e8, #8ab4f8);
        }
      }

      :root:has(> head > meta[name="color-scheme"][content="dark"]),
      html[data-theme="dark"] {
        color-scheme: dark;
      }

      :root:has(> head > meta[name="color-scheme"][content="light"]),
      html[data-theme="light"] {
        color-scheme: light;
      }
    }

    @layer base {
      :root {
        font-family: var(--font-sans);
        font-size: 1rem;
        line-height: 1.5;
        background-color: var(--bg);
        color: var(--text);
        scrollbar-color: var(--border) transparent;
        scrollbar-width: auto;
      }

      @supports not (scrollbar-color: auto) {
        ::-webkit-scrollbar {
          width: 0.75rem;
          height: 0.75rem;
        }
        ::-webkit-scrollbar-thumb {
          background-color: var(--border);
          border-radius: 0.375rem;
        }
        ::-webkit-scrollbar-track {
          background-color: transparent;
        }
      }

      body {
        min-height: 100dvh;
        display: flex;
        flex-direction: column;
        background-color: var(--bg);
        color: var(--text);
      }

      :focus-visible {
        outline: 0.125rem solid var(--focus-ring);
        outline-offset: 0.125rem;
      }

      h1, h2, h3 {
        margin: 0;
        line-height: 1.25;
      }

      h1 {
        font-size: 1.25rem;
        font-weight: 700;
      }

      h2 {
        font-size: 1rem;
        font-weight: 600;
      }

      h3 {
        font-size: 0.9rem;
        font-weight: 600;
      }

      code, pre {
        font-family: var(--font-mono);
      }

      pre {
        margin: 0;
        padding: 0.875rem;
        border-radius: 0.5rem;
        background-color: var(--code-bg);
        border: 0.0625rem solid var(--border);
        overflow-x: auto;
        font-size: 0.85rem;
        line-height: 1.45;
      }
    }

    @layer components {
      .skip-link {
        position: absolute;
        left: -9999px;
        top: 0;
        z-index: 100;
        padding: 0.625rem 1rem;
        background-color: var(--accent-blue);
        color: #ffffff;
        font-weight: 600;
        border-radius: 0.375rem;
        text-decoration: none;
      }

      .skip-link:focus {
        left: 1rem;
        top: 1rem;
      }

      .site-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.875rem 1.25rem;
        background-color: var(--surface);
        border-bottom: 0.0625rem solid var(--border);
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 0.625rem;
      }

      .brand-subtitle {
        font-size: 0.82rem;
        color: var(--text-muted);
      }

      .header-actions {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
      }

      .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        min-block-size: 2rem;
        padding: 0.25rem 0.625rem;
        border-radius: 999px;
        background-color: var(--surface-2);
        border: 0.0625rem solid var(--border);
        font-size: 0.82rem;
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
      }

      button {
        font-family: inherit;
        font-size: max(1rem, 0.9rem);
        font-weight: 600;
        line-height: 1.3;
        min-block-size: 2.375rem;
        padding: 0.45rem 0.9rem;
        border-radius: 0.5rem;
        border: 0.0625rem solid var(--border);
        background-color: var(--surface-2);
        color: var(--text);
        cursor: pointer;
      }

      button.primary {
        background-color: var(--accent-blue);
        border-color: var(--accent-blue);
        color: #ffffff;
      }

      button.success {
        background-color: var(--accent-green);
        border-color: var(--accent-green);
        color: #ffffff;
      }

      button:hover:not(:disabled) {
        filter: brightness(1.06);
      }

      button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }

      @media (pointer: coarse) {
        button, input, select {
          min-block-size: 2.75rem;
        }
      }

      .workspace-layout {
        flex: 1 1 auto;
        display: grid;
        grid-template-columns: minmax(20rem, 1.15fr) minmax(18rem, 0.85fr);
        gap: 1rem;
        padding: 1rem 1.25rem;
        max-width: 96rem;
        width: 100%;
        margin: 0 auto;
      }

      @media (max-width: 60rem) {
        .workspace-layout {
          grid-template-columns: 1fr;
        }
      }

      .panel {
        background-color: var(--surface);
        border: 0.0625rem solid var(--border);
        border-radius: 0.75rem;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.875rem;
      }

      @supports (content-visibility: auto) {
        .isolated-panel {
          content-visibility: auto;
          contain-intrinsic-size: auto 360px auto 520px;
        }
      }

      @supports not (content-visibility: auto) {
        .isolated-panel {
          contain: layout style paint;
        }
      }

      .panel-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
      }

      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
        gap: 0.625rem;
      }

      .kpi-card {
        background-color: var(--surface-2);
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
        padding: 0.625rem 0.75rem;
      }

      .kpi-label {
        font-size: 0.78rem;
        color: var(--text-muted);
      }

      .kpi-value {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.125rem;
      }

      .matrix-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        font-size: 0.82rem;
        color: var(--text-muted);
      }

      .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
      }

      .legend-swatch {
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 0.1875rem;
        display: inline-block;
      }

      .systolic-grid {
        display: grid;
        grid-template-columns: repeat(16, minmax(0, 1fr));
        gap: 0.25rem;
        padding: 0.5rem;
        background-color: var(--surface-2);
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
      }

      .mxu-cell {
        aspect-ratio: 1 / 1;
        border-radius: 0.2rem;
        background-color: var(--cell-idle);
        border: 0.0625rem solid transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font-mono);
        font-size: 0.62rem;
        color: transparent;
        transition: transform 110ms ease-out, background-color 110ms ease-out;
        overflow: hidden;
      }

      .mxu-cell[data-state="root"] {
        background-color: var(--cell-root);
        color: #ffffff;
        font-weight: 700;
        transform: scale(1.03);
      }

      .mxu-cell[data-state="accepted"] {
        background-color: var(--cell-accepted);
        color: #ffffff;
        font-weight: 700;
      }

      .mxu-cell[data-state="rejected"] {
        background-color: var(--cell-rejected);
        color: #ffffff;
      }

      .chat-transcript {
        min-height: 15rem;
        max-height: 26rem;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.625rem;
        padding: 0.75rem;
        background-color: var(--surface-2);
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
      }

      .msg-bubble {
        padding: 0.625rem 0.75rem;
        border-radius: 0.5rem;
        border: 0.0625rem solid var(--border);
        background-color: var(--surface);
        white-space: pre-wrap;
        word-break: break-word;
      }

      .msg-role {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
      }

      .controls-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: flex-end;
      }

      .field-group {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
      }

      .field-group label, .field-group legend {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-muted);
      }

      fieldset.mode-group {
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
        padding: 0.35rem 0.6rem;
        margin: 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
      }

      label.mode-option {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        border: 0.0625rem solid transparent;
        cursor: pointer;
        font-size: 0.85rem;
        color: var(--text);
      }

      label.mode-option:has(:checked) {
        background-color: var(--surface-2);
        border-color: var(--accent-blue);
        font-weight: 600;
      }

      input, textarea, select {
        font-family: inherit;
        font-size: max(1rem, 0.95rem);
        line-height: 1.45;
        color: var(--text);
        background-color: var(--surface);
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
        padding: 0.45rem 0.625rem;
      }

      textarea#prompt-input {
        width: 100%;
        height: auto;
        field-sizing: content;
        min-block-size: 3.5lh;
        max-block-size: 10lh;
        resize: vertical;
      }

      input.numeric-field {
        width: 7.5rem;
        font-family: var(--font-mono);
      }

      :is(input, textarea):is(:user-invalid, .user-invalid-fallback) {
        border-color: var(--accent-red);
      }

      input:is(:user-valid, .user-valid-fallback) {
        border-color: var(--accent-green);
      }

      .field-error {
        font-size: 0.8rem;
        color: var(--accent-red);
        min-block-size: 1.1rem;
      }

      .ab-banner {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
        gap: 0.625rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 0.0625rem solid var(--border);
        background-color: var(--surface-2);
      }

      .table-scroll {
        overflow: auto;
        container-type: inline-size;
        border: 0.0625rem solid var(--border);
        border-radius: 0.5rem;
        max-height: 15rem;
      }

      table.pulse-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 0.8rem;
        --col-2: "Accepted";
        --col-3: "Drafted";
        --col-4: "Fill";
        --col-5: "Step ms";
        --col-6: "Prefix";
      }

      table.pulse-table th,
      table.pulse-table td {
        padding: 0.4rem 0.55rem;
        border-bottom: 0.0625rem solid var(--border);
        text-align: left;
        white-space: nowrap;
      }

      table.pulse-table thead th {
        position: sticky;
        inset-block-start: 0;
        z-index: 2;
        background-color: var(--surface-2);
        font-weight: 700;
      }

      table.pulse-table tbody th {
        position: sticky;
        inset-inline-start: 0;
        z-index: 1;
        background-color: var(--surface);
        font-weight: 700;
      }

      @container (width < 36rem) {
        table.pulse-table,
        table.pulse-table thead,
        table.pulse-table tbody,
        table.pulse-table tr,
        table.pulse-table th,
        table.pulse-table td {
          display: block;
        }

        table.pulse-table thead {
          display: none;
        }

        table.pulse-table tbody tr {
          margin: 0.5rem;
          border: 0.0625rem solid var(--border);
          border-radius: 0.375rem;
          background-color: var(--surface);
        }

        table.pulse-table tbody th {
          position: sticky;
          inset-block-start: 0;
          z-index: 2;
          background-color: var(--surface-2);
        }

        table.pulse-table tbody td:nth-child(2)::before {
          content: var(--col-2) ": ";
          font-weight: 600;
          color: var(--text-muted);
        }

        table.pulse-table tbody td:nth-child(3)::before {
          content: var(--col-3) ": ";
          font-weight: 600;
          color: var(--text-muted);
        }

        table.pulse-table tbody td:nth-child(4)::before {
          content: var(--col-4) ": ";
          font-weight: 600;
          color: var(--text-muted);
        }

        table.pulse-table tbody td:nth-child(5)::before {
          content: var(--col-5) ": ";
          font-weight: 600;
          color: var(--text-muted);
        }

        table.pulse-table tbody td:nth-child(6)::before {
          content: var(--col-6) ": ";
          font-weight: 600;
          color: var(--text-muted);
        }
      }

      dialog {
        max-width: min(44rem, 92vw);
        width: 100%;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 0.0625rem solid var(--border);
        background-color: var(--surface);
        color: var(--text);
        box-shadow: 0 1.25rem 3rem rgba(0, 0, 0, 0.35);
      }

      dialog::backdrop {
        background: rgba(0, 0, 0, 0.58);
      }

      .dialog-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.75rem;
      }

      .dialog-body {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
      }

      .site-footer {
        padding: 0.75rem 1.25rem;
        border-top: 0.0625rem solid var(--border);
        font-size: 0.8rem;
        color: var(--text-muted);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.5rem;
      }
    }

    @layer utilities {
      .visually-hidden {
        position: absolute;
        width: 1px;
        height: 1px;
        margin: -1px;
        padding: 0;
        overflow: hidden;
        clip-path: inset(50%);
        white-space: nowrap;
      }

      .sr-announcers {
        position: relative;
      }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>

  <header class="site-header">
    <div class="brand">
      <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
        <rect x="2" y="2" width="10" height="10" rx="2" fill="#4285F4"></rect>
        <rect x="16" y="2" width="10" height="10" rx="2" fill="#34A853"></rect>
        <rect x="2" y="16" width="10" height="10" rx="2" fill="#FBBC04"></rect>
        <rect x="16" y="16" width="10" height="10" rx="2" fill="#EA4335"></rect>
      </svg>
      <div>
        <h1>TPUForge Serve</h1>
        <div class="brand-subtitle" id="preset-subtitle">Loading preset telemetry...</div>
      </div>
    </div>

    <div class="header-actions">
      <span class="badge-pill" id="kpi-hardware">Tile: -- | TP: -- | Quant: --</span>
      <button type="button" id="open-seed-btn">AST Context Seed</button>
      <button type="button" id="open-connect-btn">API / Colab Connect</button>
      <button type="button" id="theme-toggle-btn" aria-pressed="false" aria-label="Toggle color theme">Theme</button>
    </div>
  </header>

  <main id="main-content" class="workspace-layout" tabindex="-1">
    <div class="sr-announcers visually-hidden">
      <div id="polite-live" role="status" aria-live="polite" aria-atomic="true"></div>
      <div id="assertive-live" role="alert" aria-live="assertive" aria-atomic="true"></div>
    </div>

    <section class="panel" aria-labelledby="chat-heading">
      <div class="panel-header">
        <h2 id="chat-heading">Interactive Prompt &amp; A/B Speedup Studio</h2>
        <button type="button" id="run-ab-btn" class="success">Run A/B Speedup Benchmark</button>
      </div>

      <div id="chat-transcript" class="chat-transcript" aria-live="off">
        <div class="msg-bubble">
          <div class="msg-role">System</div>
          <div>Ready on fixed OpenXLA static HBM arena. Submit a prompt or click Run A/B Speedup Benchmark to compare Static Arena single-token decode against TPUForge Systolic Tree verification.</div>
        </div>
      </div>

      <div id="ab-result-banner" class="ab-banner" hidden>
        <div class="kpi-card">
          <div class="kpi-label">Static Arena (Baseline)</div>
          <div class="kpi-value" id="ab-static-val">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">TPUForge Tree / DFlash</div>
          <div class="kpi-value" id="ab-tree-val">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Speedup &amp; Parity</div>
          <div class="kpi-value" id="ab-speedup-val">--</div>
        </div>
      </div>

      <form id="chat-form" novalidate>
        <div class="controls-row">
          <fieldset class="mode-group">
            <legend>Execution Mode</legend>
            <label class="mode-option">
              <input type="radio" name="tpuforge-mode" value="block_tree" id="mode-block-tree">
              <span>DFlash Block Tree</span>
            </label>
            <label class="mode-option">
              <input type="radio" name="tpuforge-mode" value="tree" id="mode-tree">
              <span>AST Suffix Tree</span>
            </label>
            <label class="mode-option">
              <input type="radio" name="tpuforge-mode" value="static_arena" id="mode-static">
              <span>Static Arena (1x)</span>
            </label>
          </fieldset>

          <div class="field-group">
            <label for="max-tokens-input">Max Tokens</label>
            <input
              id="max-tokens-input"
              class="numeric-field"
              type="text"
              inputmode="numeric"
              pattern="[0-9]*"
              value="64"
              aria-invalid="false"
              data-error="max-tokens-error"
              required
            >
            <div id="max-tokens-error" class="field-error"></div>
          </div>
        </div>

        <div class="field-group" style="margin-top: 0.625rem;">
          <label for="prompt-input">Prompt (Enter sends, Shift+Enter inserts newline)</label>
          <textarea
            id="prompt-input"
            placeholder="Write a prompt or paste code to refactor using AST suffix + DFlash acceleration..."
            aria-invalid="false"
            data-error="prompt-error"
            required
          >Write a Python helper using OpenXLA static shapes.</textarea>
          <div id="prompt-error" class="field-error"></div>
        </div>

        <div class="controls-row" style="margin-top: 0.625rem; justify-content: space-between;">
          <div class="badge-pill" id="seed-status-badge">AST Seed: 0 seqs / 0 tokens</div>
          <button type="submit" class="primary" id="send-btn">Send Prompt</button>
        </div>
      </form>
    </section>

    <aside class="panel isolated-panel" aria-labelledby="telemetry-heading">
      <div class="panel-header">
        <h2 id="telemetry-heading">Live Systolic Matrix Fill</h2>
        <span class="badge-pill" id="tile-fill-badge">0 / 128 lanes</span>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Decode Speed</div>
          <div class="kpi-value" id="kpi-tokps">-- tok/s</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Accepted / Pass</div>
          <div class="kpi-value" id="kpi-tokpass">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">XLA Compiles</div>
          <div class="kpi-value" id="kpi-compiles">--</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Mean Step Latency</div>
          <div class="kpi-value" id="kpi-stepms">-- ms</div>
        </div>
      </div>

      <div class="matrix-legend" aria-label="Systolic grid color legend">
        <span class="legend-item"><span class="legend-swatch" style="background: var(--cell-root);"></span>Root Token (row 0)</span>
        <span class="legend-item"><span class="legend-swatch" style="background: var(--cell-accepted);"></span>Verified Speculative</span>
        <span class="legend-item"><span class="legend-swatch" style="background: var(--cell-rejected);"></span>Rejected Branch</span>
        <span class="legend-item"><span class="legend-swatch" style="background: var(--cell-idle);"></span>Padded Lane</span>
      </div>

      <div id="systolic-matrix" class="systolic-grid" role="img" aria-label="Systolic MXU verification tile heatmap"></div>

      <section aria-labelledby="pulse-heading">
        <h3 id="pulse-heading" style="margin-bottom: 0.4rem;">Recent Systolic Pulses</h3>
        <div class="table-scroll" tabindex="0" role="region" aria-labelledby="pulse-heading">
          <table class="pulse-table">
            <thead>
              <tr>
                <th scope="col">Step</th>
                <th scope="col">Accepted</th>
                <th scope="col">Drafted</th>
                <th scope="col">Fill</th>
                <th scope="col">Step ms</th>
                <th scope="col">Prefix</th>
              </tr>
            </thead>
            <tbody id="pulse-tbody"></tbody>
          </table>
        </div>
      </section>
    </aside>
  </main>

  <footer class="site-footer">
    <span>TPUForge OpenXLA + PJRT C ABI Speculative Runtime</span>
    <span id="footer-model-info">Zero-copy static HBM arena</span>
  </footer>

  <dialog id="ast-seed-dialog" closedby="any" aria-labelledby="ast-seed-title">
    <div class="dialog-header">
      <h2 id="ast-seed-title">Out-of-Band AST &amp; Context Seeding</h2>
      <form method="dialog">
        <button type="submit">Close</button>
      </form>
    </div>
    <div class="dialog-body">
      <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">
        Seed source code or repository paths into the C++ SuffixIndex without injecting tokens into the LLM prompt context window.
      </p>
      <div class="field-group">
        <label for="seed-path-input">Repository Directory or File Path (optional)</label>
        <input id="seed-path-input" type="text" placeholder="e.g. . or src/speculative">
      </div>
      <div class="field-group">
        <label for="seed-text-input">Raw Code / Context Snippets (optional)</label>
        <textarea id="seed-text-input" rows="4" placeholder="Paste code identifiers or imports to pre-seed SuffixIndex..."></textarea>
      </div>
      <div class="controls-row" style="justify-content: flex-end;">
        <button type="button" id="apply-seed-btn" class="primary">Update AST Seed Index</button>
      </div>
    </div>
  </dialog>

  <dialog id="connect-dialog" closedby="any" aria-labelledby="connect-title">
    <div class="dialog-header">
      <h2 id="connect-title">OpenAI-Compatible Endpoint &amp; Colab Connect</h2>
      <form method="dialog">
        <button type="submit">Close</button>
      </form>
    </div>
    <div class="dialog-body">
      <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">
        Point any OpenAI SDK client, Claude Code router, or IDE extension directly at this TPUForge Serve instance:
      </p>
      <pre role="region" aria-label="Python OpenAI SDK snippet" tabindex="0"><code id="connect-code-snippet"></code></pre>
    </div>
  </dialog>

  <script id="tpuforge-initial-state" type="application/json">__INITIAL_STATE_JSON__</script>
  <script>
    (function () {
      var rawStateEl = document.getElementById("tpuforge-initial-state");
      var state = {};
      try {
        state = JSON.parse(rawStateEl.textContent || "{}");
      } catch (_err) {
        state = {};
      }

      var tileSize = Number(state.tile_size) || 128;
      var hasDraft = Boolean(state.has_draft_model);
      var politeLive = document.getElementById("polite-live");
      var assertiveLive = document.getElementById("assertive-live");

      function announcePolite(msg) {
        politeLive.textContent = msg;
      }

      function announceError(msg) {
        assertiveLive.textContent = msg;
      }

      var matrixEl = document.getElementById("systolic-matrix");
      var cells = [];
      function buildMatrix(count) {
        matrixEl.textContent = "";
        cells = [];
        for (var i = 0; i < count; i++) {
          var cell = document.createElement("div");
          cell.className = "mxu-cell";
          cell.setAttribute("data-state", "idle");
          cell.title = "Lane " + i;
          matrixEl.appendChild(cell);
          cells.push(cell);
        }
      }
      buildMatrix(tileSize);

      function updateStatusUI(st) {
        if (!st) {
          return;
        }
        tileSize = Number(st.tile_size) || tileSize;
        if (cells.length !== tileSize) {
          buildMatrix(tileSize);
        }
        hasDraft = Boolean(st.has_draft_model);
        var modeBlockBtn = document.getElementById("mode-block-tree");
        modeBlockBtn.disabled = !hasDraft;
        var activeDefault = st.default_mode || (hasDraft ? "block_tree" : "tree");
        var radio = document.querySelector('input[name="tpuforge-mode"][value="' + activeDefault + '"]');
        if (radio && !document.querySelector('input[name="tpuforge-mode"]:checked')) {
          radio.checked = true;
        }
        if (!document.querySelector('input[name="tpuforge-mode"]:checked')) {
          document.getElementById("mode-tree").checked = true;
        }

        document.getElementById("preset-subtitle").textContent =
          (st.preset || st.model_id || "custom") + " (" + (st.hardware_label || "OpenXLA") + ")";
        document.getElementById("kpi-hardware").textContent =
          "Tile: " + tileSize + " | TP: " + (st.tp_size || 1) + " | Quant: " + (st.weight_quant || "none");
        document.getElementById("seed-status-badge").textContent =
          "AST Seed: " + (st.ast_seed_sequences || 0) + " seqs / " + (st.ast_seed_tokens || 0) + " tokens";
        document.getElementById("footer-model-info").textContent =
          "Model: " + (st.model_id || "unknown") +
          (st.draft_model_id ? " | Draft: " + st.draft_model_id : "") +
          " | Max seq: " + (st.max_seq_len || "--");
        document.getElementById("tile-fill-badge").textContent = "0 / " + tileSize + " lanes";

        var baseUrl = window.location.origin + "/v1";
        document.getElementById("connect-code-snippet").textContent =
          "from openai import OpenAI\\n\\n" +
          "client = OpenAI(base_url=\\\"" + baseUrl + "\\\", api_key=\\\"tpuforge\\\")\\n" +
          "resp = client.chat.completions.create(\\n" +
          "    model=\\\"" + (st.preset || st.model_id || "tpuforge") + "\\\",\\n" +
          "    messages=[{\\\"role\\\": \\\"user\\\", \\\"content\\\": \\\"Hello from TPUForge!\\\"}],\\n" +
          "    max_tokens=128,\\n" +
          "    stream=True,\\n" +
          ")\\n" +
          "for chunk in resp:\\n" +
          "    print(chunk.choices[0].delta.content or \\\"\\\", end=\\\"\\\", flush=True)";
      }

      updateStatusUI(state);

      function renderPulse(pulse) {
        if (!pulse) {
          return;
        }
        var acceptedRows = new Set(pulse.accepted_rows || []);
        var draftedNodes = Number(pulse.drafted_nodes) || 0;
        for (var i = 0; i < cells.length; i++) {
          var cell = cells[i];
          if (i === 0 && acceptedRows.has(0)) {
            cell.setAttribute("data-state", "root");
            cell.textContent = "0";
          } else if (acceptedRows.has(i)) {
            cell.setAttribute("data-state", "accepted");
            cell.textContent = String(i);
          } else if (i < draftedNodes) {
            cell.setAttribute("data-state", "rejected");
            cell.textContent = "";
          } else {
            cell.setAttribute("data-state", "idle");
            cell.textContent = "";
          }
        }
        document.getElementById("tile-fill-badge").textContent =
          draftedNodes + " / " + tileSize + " lanes (" + Math.round((draftedNodes / tileSize) * 100) + "%)";
      }

      function appendPulseRow(tbody, pulse) {
        var draftedNodes = Number(pulse.drafted_nodes) || 0;
        var tr = document.createElement("tr");
        var rowHdr = document.createElement("th");
        rowHdr.setAttribute("scope", "row");
        rowHdr.textContent = String(pulse.step_index || 0);
        tr.appendChild(rowHdr);

        var cols = [
          String((pulse.accepted_rows || []).length),
          String(draftedNodes),
          Math.round((draftedNodes / tileSize) * 100) + "%",
          Number(pulse.step_ms || 0).toFixed(2),
          String(pulse.prefix_len || 0)
        ];
        for (var c = 0; c < cols.length; c++) {
          var td = document.createElement("td");
          td.textContent = cols[c];
          tr.appendChild(td);
        }
        tbody.insertBefore(tr, tbody.firstChild);
      }

      var pendingPulse = null;
      var pendingPulseRows = [];
      var pulseRafId = 0;
      var pulseRafUsesTimeout = false;

      function flushPulseFrame() {
        pulseRafId = 0;
        pulseRafUsesTimeout = false;
        if (pendingPulse) {
          var latest = pendingPulse;
          pendingPulse = null;
          renderPulse(latest);
        }
        if (pendingPulseRows.length > 0) {
          var rows = pendingPulseRows;
          pendingPulseRows = [];
          var tbody = document.getElementById("pulse-tbody");
          var startIdx = rows.length > 32 ? rows.length - 32 : 0;
          for (var r = startIdx; r < rows.length; r++) {
            appendPulseRow(tbody, rows[r]);
          }
          while (tbody.children.length > 32) {
            tbody.removeChild(tbody.lastChild);
          }
        }
      }

      function resetPulseQueue() {
        if (pulseRafId) {
          if (!pulseRafUsesTimeout && typeof window.cancelAnimationFrame === "function") {
            window.cancelAnimationFrame(pulseRafId);
          } else {
            clearTimeout(pulseRafId);
          }
          pulseRafId = 0;
          pulseRafUsesTimeout = false;
        }
        pendingPulse = null;
        pendingPulseRows = [];
      }

      function enqueuePulse(pulse) {
        if (!pulse) {
          return;
        }
        pendingPulse = pulse;
        pendingPulseRows.push(pulse);
        if (pendingPulseRows.length > 64) {
          pendingPulseRows = pendingPulseRows.slice(-32);
        }
        if (!pulseRafId) {
          if (typeof window.requestAnimationFrame === "function") {
            pulseRafUsesTimeout = false;
            pulseRafId = window.requestAnimationFrame(flushPulseFrame);
          } else {
            pulseRafUsesTimeout = true;
            pulseRafId = setTimeout(flushPulseFrame, 0);
          }
        }
      }

      function applyFinalTelemetry(t) {
        if (!t) {
          return;
        }
        document.getElementById("kpi-tokps").textContent = Number(t.decode_tok_per_s || 0).toFixed(1) + " tok/s";
        document.getElementById("kpi-tokpass").textContent = Number(t.mean_tokens_per_decode_pass || 0).toFixed(2);
        var deltaComp = Number(t.compile_count || 0);
        document.getElementById("kpi-compiles").textContent = deltaComp + " (total " + (t.total_compiles || 0) + ")";
        var avgMs = t.decode_forward_passes ? (Number(t.decode_ms || 0) / Number(t.decode_forward_passes)) : 0;
        document.getElementById("kpi-stepms").textContent = avgMs.toFixed(2) + " ms";
      }

      function appendMessage(roleLabel, initialText) {
        var box = document.getElementById("chat-transcript");
        var bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        var roleEl = document.createElement("div");
        roleEl.className = "msg-role";
        roleEl.textContent = roleLabel;
        var bodyEl = document.createElement("div");
        bodyEl.textContent = initialText;
        bubble.appendChild(roleEl);
        bubble.appendChild(bodyEl);
        box.appendChild(bubble);
        box.scrollTop = box.scrollHeight;
        return bodyEl;
      }

      function selectedMode() {
        var checked = document.querySelector('input[name="tpuforge-mode"]:checked');
        return checked ? checked.value : "tree";
      }

      var maxTokensInput = document.getElementById("max-tokens-input");
      var maxTokensErr = document.getElementById("max-tokens-error");
      var promptInput = document.getElementById("prompt-input");
      var promptErr = document.getElementById("prompt-error");
      var sendBtn = document.getElementById("send-btn");
      var abBtn = document.getElementById("run-ab-btn");
      var userInvalidDirty = new WeakMap();
      var supportsUserInvalid = typeof CSS !== "undefined" && typeof CSS.supports === "function" && CSS.supports("selector(:user-invalid)");
      var fieldAnnounceTimer = 0;
      var announcedField = null;

      function linkFieldError(el, shown) {
        var errId = el.getAttribute("data-error");
        if (!errId) {
          return;
        }
        var raw = el.getAttribute("aria-describedby") || "";
        var ids = raw.split(/\\s+/).filter(function (id) { return id && id !== errId; });
        if (shown) {
          ids.push(errId);
        }
        if (ids.length > 0) {
          el.setAttribute("aria-describedby", ids.join(" "));
        } else {
          el.removeAttribute("aria-describedby");
        }
      }

      function scheduleFieldAnnouncement(el) {
        var errId = el.getAttribute("data-error");
        var errEl = errId ? document.getElementById(errId) : null;
        if (!errEl || !errEl.textContent.trim()) {
          return;
        }
        clearTimeout(fieldAnnounceTimer);
        announcedField = el;
        var msg = errEl.textContent.trim();
        fieldAnnounceTimer = setTimeout(function () {
          announcePolite(msg);
        }, 500);
      }

      function syncFieldAria(el, valid, shouldAnnounce) {
        el.setAttribute("aria-invalid", valid ? "false" : "true");
        linkFieldError(el, !valid);
        var track = userInvalidDirty.get(el) || { hasInteracted: false, hasBlurred: false };
        var showFallback = !supportsUserInvalid && (track.hasInteracted && track.hasBlurred);
        el.classList.toggle("user-invalid-fallback", showFallback && !valid);
        el.classList.toggle("user-valid-fallback", showFallback && valid);
        if (valid && announcedField === el) {
          clearTimeout(fieldAnnounceTimer);
          announcedField = null;
        } else if (!valid && shouldAnnounce) {
          scheduleFieldAnnouncement(el);
        }
      }

      function validateMaxTokens(shouldAnnounce) {
        var raw = maxTokensInput.value.trim();
        var num = Number(raw);
        if (!raw || !/^[0-9]+$/.test(raw) || num < 1 || num > 4096) {
          maxTokensInput.setCustomValidity("Enter an integer between 1 and 4096.");
          maxTokensErr.textContent = "Enter 1 to 4096.";
          syncFieldAria(maxTokensInput, false, Boolean(shouldAnnounce));
          return false;
        }
        maxTokensInput.setCustomValidity("");
        maxTokensErr.textContent = "";
        syncFieldAria(maxTokensInput, true, false);
        return true;
      }

      function validatePrompt(shouldAnnounce) {
        if (!promptInput.value.trim()) {
          promptInput.setCustomValidity("Prompt cannot be empty.");
          promptErr.textContent = "Prompt cannot be empty.";
          syncFieldAria(promptInput, false, Boolean(shouldAnnounce));
          return false;
        }
        promptInput.setCustomValidity("");
        promptErr.textContent = "";
        syncFieldAria(promptInput, true, false);
        return true;
      }

      maxTokensInput.addEventListener("blur", function () {
        var stTrack = userInvalidDirty.get(maxTokensInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        stTrack.hasBlurred = true;
        userInvalidDirty.set(maxTokensInput, stTrack);
        validateMaxTokens(true);
      });
      maxTokensInput.addEventListener("input", function () {
        var stTrack = userInvalidDirty.get(maxTokensInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        userInvalidDirty.set(maxTokensInput, stTrack);
        if (stTrack.hasBlurred || /^[0-9]+$/.test(maxTokensInput.value.trim())) {
          validateMaxTokens(false);
        }
      });
      maxTokensInput.addEventListener("change", function () {
        var stTrack = userInvalidDirty.get(maxTokensInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        stTrack.hasBlurred = true;
        userInvalidDirty.set(maxTokensInput, stTrack);
        validateMaxTokens(true);
      });

      promptInput.addEventListener("blur", function () {
        var stTrack = userInvalidDirty.get(promptInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        stTrack.hasBlurred = true;
        userInvalidDirty.set(promptInput, stTrack);
        validatePrompt(true);
      });
      promptInput.addEventListener("input", function () {
        var stTrack = userInvalidDirty.get(promptInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        userInvalidDirty.set(promptInput, stTrack);
        if (stTrack.hasBlurred || promptInput.value.trim()) {
          validatePrompt(false);
        }
      });
      promptInput.addEventListener("change", function () {
        var stTrack = userInvalidDirty.get(promptInput) || { hasInteracted: false, hasBlurred: false };
        stTrack.hasInteracted = true;
        stTrack.hasBlurred = true;
        userInvalidDirty.set(promptInput, stTrack);
        validatePrompt(true);
      });

      var isBusy = false;
      function setBusy(busy) {
        isBusy = busy;
        sendBtn.disabled = busy;
        abBtn.disabled = busy;
      }

      async function streamCompletion(promptText, maxTokens, mode, targetEl) {
        var res = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: state.preset || state.model_id || "tpuforge",
            messages: [{ role: "user", content: promptText }],
            max_tokens: maxTokens,
            temperature: 0.0,
            mode: mode,
            stream: true
          })
        });
        if (!res.ok) {
          var errJson = await res.json().catch(function () { return {}; });
          throw new Error((errJson.error && errJson.error.message) || ("HTTP " + res.status));
        }

        var reader = res.body.getReader();
        var decoder = new TextDecoder("utf-8");
        var buffer = "";
        var finalTelemetry = null;
        var fullText = "";

        try {
          while (true) {
            var part = await reader.read();
            if (part.done) {
              break;
            }
            buffer += decoder.decode(part.value, { stream: true });
            var lines = buffer.split("\\n");
            buffer = lines.pop() || "";
            for (var i = 0; i < lines.length; i++) {
              var line = lines[i].trim();
              if (!line.startsWith("data:")) {
                continue;
              }
              var payloadStr = line.slice(5).trim();
              if (payloadStr === "[DONE]") {
                continue;
              }
              var evt = JSON.parse(payloadStr);
              if (evt.error) {
                throw new Error((evt.error && evt.error.message) || "Streaming engine error");
              }
              var delta = evt.choices && evt.choices[0] && evt.choices[0].delta && evt.choices[0].delta.content;
              if (delta && targetEl) {
                fullText += delta;
                targetEl.textContent = fullText;
              }
              if (evt.tpuforge_step) {
                enqueuePulse(evt.tpuforge_step);
              }
              if (evt.tpuforge_telemetry) {
                finalTelemetry = evt.tpuforge_telemetry;
                applyFinalTelemetry(finalTelemetry);
              }
            }
          }
        } finally {
          flushPulseFrame();
        }
        return { text: fullText, telemetry: finalTelemetry };
      }

      document.getElementById("chat-form").addEventListener("submit", async function (event) {
        event.preventDefault();
        if (isBusy) {
          return;
        }
        userInvalidDirty.set(maxTokensInput, { hasInteracted: true, hasBlurred: true });
        userInvalidDirty.set(promptInput, { hasInteracted: true, hasBlurred: true });
        var okTokens = validateMaxTokens(false);
        var okPrompt = validatePrompt(false);
        if (!okTokens) {
          maxTokensInput.focus();
          return;
        }
        if (!okPrompt) {
          promptInput.focus();
          return;
        }

        var mode = selectedMode();
        var maxToks = Number(maxTokensInput.value.trim());
        var text = promptInput.value.trim();
        appendMessage("User (" + mode + ")", text);
        var assistantBody = appendMessage("Model (" + mode + ")", "");
        resetPulseQueue();
        document.getElementById("pulse-tbody").textContent = "";

        setBusy(true);
        announcePolite("Generating completion on TPUForge engine.");
        try {
          var res = await streamCompletion(text, maxToks, mode, assistantBody);
          announcePolite(
            "Generation finished at " +
            Number((res.telemetry && res.telemetry.decode_tok_per_s) || 0).toFixed(1) +
            " tokens per second."
          );
        } catch (err) {
          assistantBody.textContent = "Error: " + (err.message || String(err));
          announceError("Generation failed: " + (err.message || String(err)));
        } finally {
          setBusy(false);
        }
      });

      promptInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
          if (event.isComposing || event.keyCode === 229) {
            return;
          }
          event.preventDefault();
          document.getElementById("chat-form").requestSubmit();
        }
      });

      abBtn.addEventListener("click", async function () {
        if (isBusy) {
          return;
        }
        userInvalidDirty.set(maxTokensInput, { hasInteracted: true, hasBlurred: true });
        userInvalidDirty.set(promptInput, { hasInteracted: true, hasBlurred: true });
        var okTokens = validateMaxTokens(false);
        var okPrompt = validatePrompt(false);
        if (!okTokens || !okPrompt) {
          (!okTokens ? maxTokensInput : promptInput).focus();
          return;
        }

        var maxToks = Number(maxTokensInput.value.trim());
        var promptText = promptInput.value.trim();
        var fastMode = hasDraft ? "block_tree" : "tree";
        var banner = document.getElementById("ab-result-banner");
        banner.hidden = false;
        document.getElementById("ab-static-val").textContent = "Running...";
        document.getElementById("ab-tree-val").textContent = "Queued...";
        document.getElementById("ab-speedup-val").textContent = "--";

        setBusy(true);
        announcePolite("Running A/B benchmark between Static Arena and TPUForge Speculative mode.");
        try {
          resetPulseQueue();
          document.getElementById("pulse-tbody").textContent = "";
          var staticBody = appendMessage("A/B Static Arena", "");
          var aRes = await streamCompletion(promptText, maxToks, "static_arena", staticBody);
          var aTps = Number((aRes.telemetry && aRes.telemetry.decode_tok_per_s) || 0);
          document.getElementById("ab-static-val").textContent = aTps.toFixed(1) + " tok/s";

          resetPulseQueue();
          document.getElementById("pulse-tbody").textContent = "";
          var fastBody = appendMessage("A/B " + fastMode, "");
          var bRes = await streamCompletion(promptText, maxToks, fastMode, fastBody);
          var bTps = Number((bRes.telemetry && bRes.telemetry.decode_tok_per_s) || 0);
          document.getElementById("ab-tree-val").textContent = bTps.toFixed(1) + " tok/s";

          var speedup = aTps > 0 ? (bTps / aTps) : 0;
          var parity = aRes.text === bRes.text ? "Exact Match" : "Mismatch";
          document.getElementById("ab-speedup-val").textContent =
            speedup.toFixed(2) + "x (" + parity + ")";
          announcePolite("A/B benchmark complete: " + speedup.toFixed(2) + "x speedup, " + parity + ".");
        } catch (err) {
          announceError("A/B benchmark failed: " + (err.message || String(err)));
        } finally {
          setBusy(false);
        }
      });

      function setupDialog(dialogId, triggerId) {
        var dlg = document.getElementById(dialogId);
        var btn = document.getElementById(triggerId);
        btn.addEventListener("click", function () {
          dlg.showModal();
        });
        if (!("closedBy" in HTMLDialogElement.prototype)) {
          dlg.addEventListener("click", function (event) {
            var rect = dlg.getBoundingClientRect();
            var inside =
              event.clientX >= rect.left &&
              event.clientX <= rect.right &&
              event.clientY >= rect.top &&
              event.clientY <= rect.bottom;
            if (!inside && event.target === dlg) {
              dlg.close();
            }
          });
        }
      }
      setupDialog("ast-seed-dialog", "open-seed-btn");
      setupDialog("connect-dialog", "open-connect-btn");

      document.getElementById("apply-seed-btn").addEventListener("click", async function () {
        var pathVal = document.getElementById("seed-path-input").value.trim();
        var textVal = document.getElementById("seed-text-input").value.trim();
        var lines = textVal
          ? textVal.split(/\\r?\\n/).map(function (s) { return s.trim(); }).filter(Boolean)
          : [];
        try {
          var res = await fetch("/api/seed", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ast_seed_path: pathVal || null,
              seed_texts: lines
            })
          });
          var data = await res.json();
          if (!res.ok) {
            throw new Error((data.error && data.error.message) || ("HTTP " + res.status));
          }
          updateStatusUI(data);
          document.getElementById("ast-seed-dialog").close();
          announcePolite("Updated AST SuffixIndex seed corpus.");
        } catch (err) {
          announceError("Failed to update AST seed: " + (err.message || String(err)));
        }
      });

      var themeBtn = document.getElementById("theme-toggle-btn");
      var colorSchemeMeta = document.querySelector('meta[name="color-scheme"]');

      function currentSystemScheme() {
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
      }

      function effectiveTheme() {
        var attr = document.documentElement.getAttribute("data-theme");
        if (attr === "dark" || attr === "light") {
          return attr;
        }
        return currentSystemScheme();
      }

      function syncThemeControl() {
        var active = effectiveTheme();
        themeBtn.setAttribute("aria-pressed", active === "dark" ? "true" : "false");
        themeBtn.textContent = "Theme: " + active;
      }

      syncThemeControl();

      window.addEventListener("storage", function (event) {
        if (event.key !== "tpuforge-theme") {
          return;
        }
        var nextVal = event.newValue;
        if (nextVal === "dark" || nextVal === "light") {
          document.documentElement.setAttribute("data-theme", nextVal);
          if (colorSchemeMeta) {
            colorSchemeMeta.content = nextVal;
          }
        } else {
          document.documentElement.removeAttribute("data-theme");
          if (colorSchemeMeta) {
            colorSchemeMeta.content = "light dark";
          }
        }
        syncThemeControl();
      });

      themeBtn.addEventListener("click", function () {
        var rendered = effectiveTheme();
        var next = rendered === "dark" ? "light" : "dark";
        var osScheme = currentSystemScheme();
        try {
          if (next === osScheme) {
            if (window.localStorage) {
              window.localStorage.removeItem("tpuforge-theme");
            }
            document.documentElement.removeAttribute("data-theme");
            if (colorSchemeMeta) {
              colorSchemeMeta.content = "light dark";
            }
          } else {
            if (window.localStorage) {
              window.localStorage.setItem("tpuforge-theme", next);
            }
            document.documentElement.setAttribute("data-theme", next);
            if (colorSchemeMeta) {
              colorSchemeMeta.content = next;
            }
          }
        } catch (_err) {
          document.documentElement.setAttribute("data-theme", next);
          if (colorSchemeMeta) {
            colorSchemeMeta.content = next;
          }
        }
        syncThemeControl();
      });
    })();
  </script>
</body>
</html>
"""


def render_hud_html(initial_status: dict[str, Any]) -> str:
    """Render the self-contained TPUForge Serve Live Systolic HUD HTML page."""
    safe_json = json.dumps(initial_status).replace("</", "<\\/")
    return _HUD_HTML_TEMPLATE.replace("__INITIAL_STATE_JSON__", safe_json)
