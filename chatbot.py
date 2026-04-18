#!/usr/bin/env python3
"""
chatbot.py — Pocket-Agent Gradio demo.

Features
--------
- Multi-turn conversation with history
- Tool-call output shown in a separate highlighted panel
- Runs on Colab CPU runtime  (loads quantized GGUF model)
- No network calls at inference

Run: python chatbot.py
"""

import json, re
import gradio as gr
from inference import run   # reuse the grader interface

# ── Helpers ───────────────────────────────────────────────────────────
_TOOL_COLORS = {
    "weather":  "#4A90D9",
    "calendar": "#7B68EE",
    "convert":  "#50C878",
    "currency": "#FFB347",
    "sql":      "#FF6B6B",
}

def _parse_tool(text: str):
    """Return (tool_name, args_dict) or None."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        return obj.get("tool"), obj.get("args", {})
    except Exception:
        return None


def _format_tool_html(tool_name: str, args: dict) -> str:
    color = _TOOL_COLORS.get(tool_name, "#888")
    args_str = json.dumps(args, indent=2, ensure_ascii=False)
    return (
        f'<div style="background:#1e1e1e;border-left:4px solid {color};'
        f'padding:12px;border-radius:6px;font-family:monospace;color:#f8f8f2;">'
        f'<span style="color:{color};font-weight:bold;">🔧 {tool_name.upper()}</span><br/>'
        f'<pre style="margin:8px 0 0 0;color:#a8ff78;">{args_str}</pre>'
        f"</div>"
    )


# ── Chat logic ────────────────────────────────────────────────────────
def respond(user_msg: str, chat_history: list, tool_display: str):
    """Called on each user message."""
    if not user_msg.strip():
        return chat_history, tool_display, ""

    # Build history in grader format from chat_history tuples
    history_dicts = []
    for human, assistant in chat_history:
        history_dicts.append({"role": "user",      "content": human})
        history_dicts.append({"role": "assistant",  "content": assistant})

    # Get model response
    response = run(user_msg, history_dicts)

    # Parse tool call for display
    parsed = _parse_tool(response)
    if parsed:
        tool_name, args = parsed
        new_tool_html = _format_tool_html(tool_name, args)
        display_response = f"`{response}`"   # show raw in chat too
    else:
        new_tool_html = (
            '<div style="background:#2d2d2d;padding:12px;border-radius:6px;'
            'color:#aaa;font-style:italic;">No tool call — plain response</div>'
        )
        display_response = response

    chat_history = chat_history + [(user_msg, display_response)]
    return chat_history, new_tool_html, ""


def clear_chat():
    return [], '<div style="color:#aaa;font-style:italic;padding:8px;">Tool calls will appear here</div>', ""


# ── UI ────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What's the weather in London?",
    "Convert 100 USD to EUR",
    "5 miles to kilometers",
    "Schedule Team Sync on 2025-06-10",
    "Show all active orders from the database",
    "How are you?",
    "Send an email to John",
]

with gr.Blocks(
    title="Pocket-Agent Demo",
    theme=gr.themes.Base(primary_hue="blue"),
    css=".tool-panel { min-height: 120px; }",
) as demo:
    gr.Markdown(
        "# 🤖 Pocket-Agent\n"
        "**On-device tool-calling assistant** · "
        "Supports: `weather` · `calendar` · `convert` · `currency` · `sql`\n\n"
        "*Try typing a request or click an example below.*"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                bubble_full_width=False,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message…",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                )
                send_btn  = gr.Button("Send ▶", scale=1, variant="primary")
                clear_btn = gr.Button("Clear 🗑", scale=1)

            gr.Examples(
                examples=EXAMPLES,
                inputs=user_input,
                label="Example prompts",
            )

        with gr.Column(scale=2):
            tool_output = gr.HTML(
                value='<div style="color:#aaa;font-style:italic;padding:8px;">Tool calls will appear here</div>',
                label="Tool Call Output",
                elem_classes="tool-panel",
            )
            gr.Markdown(
                "**Tool colours:**\n"
                "- 🔵 Weather  🟣 Calendar\n"
                "- 🟢 Convert  🟠 Currency  🔴 SQL"
            )

    # Events
    send_btn.click(
        respond,
        inputs=[user_input, chatbot, tool_output],
        outputs=[chatbot, tool_output, user_input],
    )
    user_input.submit(
        respond,
        inputs=[user_input, chatbot, tool_output],
        outputs=[chatbot, tool_output, user_input],
    )
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, tool_output, user_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # creates a public link on Colab
        debug=False,
    )
