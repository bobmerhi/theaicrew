"""AI Crew — a multi-provider chat portal with a crew of AI agents.

Providers (all via OpenAI-compatible endpoints): NVIDIA NIM, Groq, OpenRouter, Google AI Studio.
Run locally:  streamlit run app.py
"""

import json
import os
import uuid

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Crew", page_icon="🧑‍🚀", layout="wide")

# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "NVIDIA NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "secret": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "secret": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "Google AI Studio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
}

# --------------------------------------------------------------------------- #
# Default crew
# --------------------------------------------------------------------------- #


def new_agent(name, system_prompt, provider, model=None, temperature=0.4, enabled=True):
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "system_prompt": system_prompt,
        "provider": provider,
        "model": model or PROVIDERS[provider]["default_model"],
        "temperature": temperature,
        "enabled": enabled,
    }


def default_crew():
    return [
        new_agent(
            "Manager",
            "You are the Manager of an AI crew. Break the user's goal into a clear numbered plan, "
            "assign each step to a crew member by name, and keep the team on track. When asked to "
            "summarize, merge all outputs into one concise final deliverable and list open issues.",
            "Groq",
        ),
        new_agent(
            "Researcher",
            "You are the Researcher. Gather and organize relevant facts, options, trade-offs and "
            "pitfalls. Be explicit about uncertainty. Never invent sources or citations.",
            "Google AI Studio",
        ),
        new_agent(
            "Coder",
            "You are the Coder. Write clean, working, minimal code with brief explanations. "
            "State your assumptions. Use fenced code blocks.",
            "NVIDIA NIM",
            temperature=0.2,
        ),
        new_agent(
            "Builder",
            "You are the Builder. Turn plans and code into a runnable project: file structure, "
            "dependencies, setup and run commands, configuration and deployment steps.",
            "OpenRouter",
            temperature=0.2,
        ),
        new_agent(
            "Reviewer",
            "You are the Reviewer. Critically review the work for bugs, security issues, missing "
            "requirements and unclear parts. Give a prioritized list of concrete fixes and a verdict.",
            "Groq",
            temperature=0.2,
        ),
    ]


if "agents" not in st.session_state:
    st.session_state.agents = default_crew()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "last_run" not in st.session_state:
    st.session_state.last_run = None

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def get_key(provider: str) -> str:
    """Key priority: typed in sidebar > Streamlit secrets > environment."""
    names = [PROVIDERS[provider]["secret"]]
    if provider == "Google AI Studio":
        names.append("GEMINI_API_KEY")  # alias used by the earlier app
    typed = st.session_state.api_keys.get(provider, "").strip()
    if typed:
        return typed
    for name in names:
        try:
            if name in st.secrets:
                return str(st.secrets[name]).strip()
        except Exception:
            pass
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    return ""


def stream_completion(agent, messages, placeholder) -> str:
    """Stream a chat completion into `placeholder` and return the full text."""
    provider = agent["provider"]
    key = get_key(provider)
    if not key:
        raise ValueError(f"No API key for {provider}. Add it in the sidebar or in Streamlit secrets.")
    headers = {"X-Title": "AI Crew"} if provider == "OpenRouter" else None
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
        default_headers=headers,
    )
    stream = client.chat.completions.create(
        model=agent["model"],
        messages=messages,
        temperature=agent.get("temperature", 0.4),
        stream=True,
    )
    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            text += delta
            placeholder.markdown(text + "▌")
    placeholder.markdown(text)
    return text


def clip(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def enabled_agents():
    return [a for a in st.session_state.agents if a["enabled"]]


def agent_by_id(agent_id):
    return next((a for a in st.session_state.agents if a["id"] == agent_id), None)


# --------------------------------------------------------------------------- #
# Sidebar: keys + bulk provider switch
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧑‍🚀 AI Crew")
    st.subheader("API keys")
    for prov, cfg in PROVIDERS.items():
        typed = st.text_input(
            prov,
            type="password",
            key=f"key_{prov}",
            placeholder="paste key (session only)",
        )
        st.session_state.api_keys[prov] = typed
        st.caption("✅ key found" if get_key(prov) else "⚠️ no key")

    st.divider()
    st.subheader("Quick setup")
    one = st.selectbox("Run the whole crew on one provider", list(PROVIDERS))
    if st.button("Apply to all agents", use_container_width=True):
        for a in st.session_state.agents:
            a["provider"] = one
            a["model"] = PROVIDERS[one]["default_model"]
            st.session_state[f"p_{a['id']}"] = one
            st.session_state[f"m_{a['id']}"] = a["model"]
        st.rerun()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_chat, tab_crew, tab_manage = st.tabs(["💬 Chat", "🚀 Crew run", "👥 Manage crew"])

# ----- Chat with a single agent --------------------------------------------- #
with tab_chat:
    agents = enabled_agents()
    if not agents:
        st.info("Enable at least one agent in **Manage crew**.")
    else:
        chosen = st.selectbox(
            "Talk to",
            agents,
            format_func=lambda a: f"{a['name']} · {a['provider']} / {a['model']}",
            key="chat_agent",
        )
        history = st.session_state.chats.setdefault(chosen["id"], [])

        top = st.columns([6, 1])
        with top[1]:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chats[chosen["id"]] = []
                st.rerun()

        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input(f"Message {chosen['name']}…")
        if prompt:
            history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            messages = [{"role": "system", "content": chosen["system_prompt"]}] + history
            with st.chat_message("assistant"):
                ph = st.empty()
                try:
                    reply = stream_completion(chosen, messages, ph)
                    history.append({"role": "assistant", "content": reply})
                except Exception as e:  # noqa: BLE001
                    ph.empty()
                    history.pop()  # drop the unanswered user turn
                    st.error(f"{type(e).__name__}: {e}")

# ----- Crew pipeline --------------------------------------------------------- #
with tab_crew:
    agents = enabled_agents()
    if not agents:
        st.info("Enable at least one agent in **Manage crew**.")
    else:
        st.write(
            "Give the crew a goal. The Manager plans, each selected agent works in order "
            "(seeing the plan and earlier outputs), then the Manager writes the final summary."
        )
        task = st.text_area("Goal / task", height=120, key="crew_task",
                            placeholder="e.g. Build a Streamlit page that tracks my daily habits")

        manager_options = [None] + agents
        default_mgr = next((a for a in agents if a["name"].lower() == "manager"), None)
        manager = st.selectbox(
            "Manager (plans first, summarizes last)",
            manager_options,
            index=manager_options.index(default_mgr) if default_mgr else 0,
            format_func=lambda a: "— none —" if a is None else a["name"],
            key="crew_manager",
        )

        ids = [a["id"] for a in agents]
        default_steps = [a["id"] for a in agents if not (manager and a["id"] == manager["id"])]
        steps_ids = st.multiselect(
            "Working order",
            ids,
            default=default_steps,
            format_func=lambda i: agent_by_id(i)["name"],
            key=f"crew_steps_{hash(tuple(ids))}_{manager['id'] if manager else 'none'}",
        )

        if st.button("🚀 Run crew", type="primary", disabled=not task.strip()):
            transcript = [("Task", task.strip())]
            plan = ""
            outputs = []

            def run_step(agent, title, instruction):
                with st.expander(f"{title} — {agent['name']} ({agent['provider']} / {agent['model']})",
                                 expanded=True):
                    ph = st.empty()
                    try:
                        msgs = [
                            {"role": "system", "content": agent["system_prompt"]},
                            {"role": "user", "content": instruction},
                        ]
                        text = stream_completion(agent, msgs, ph)
                    except Exception as e:  # noqa: BLE001
                        ph.empty()
                        text = f"[failed: {type(e).__name__}: {e}]"
                        st.error(text)
                transcript.append((f"{title} — {agent['name']}", text))
                return text

            goal = task.strip()
            if manager:
                plan = run_step(
                    manager,
                    "Plan",
                    f"Goal:\n{goal}\n\nCrew members: "
                    + ", ".join(a["name"] for a in agents if a["id"] in steps_ids)
                    + "\n\nWrite a numbered plan and say who does what.",
                )

            for sid in steps_ids:
                agent = agent_by_id(sid)
                context = f"Goal:\n{goal}\n"
                if plan:
                    context += f"\nManager's plan:\n{clip(plan)}\n"
                for name, out in outputs:
                    context += f"\nOutput from {name}:\n{clip(out)}\n"
                text = run_step(
                    agent,
                    f"Step {len(outputs) + 1}",
                    context + f"\nYou are {agent['name']}. Do your part now.",
                )
                outputs.append((agent["name"], text))

            if manager and outputs:
                ctx = f"Goal:\n{goal}\n"
                for name, out in outputs:
                    ctx += f"\nOutput from {name}:\n{clip(out)}\n"
                run_step(
                    manager,
                    "Final summary",
                    ctx + "\nMerge everything into one final deliverable. List open issues at the end.",
                )

            st.session_state.last_run = "\n\n".join(f"## {t}\n\n{b}" for t, b in transcript)

        if st.session_state.last_run:
            st.download_button(
                "⬇️ Download last run (Markdown)",
                st.session_state.last_run,
                file_name="crew_run.md",
                mime="text/markdown",
            )

# ----- Manage crew ----------------------------------------------------------- #
with tab_manage:
    st.write("Edit agents, switch providers/models, or add new positions.")
    for a in list(st.session_state.agents):
        aid = a["id"]
        icon = "🟢" if a["enabled"] else "⚪"
        with st.expander(f"{icon} {a['name']} — {a['provider']} / {a['model']}"):
            a["name"] = st.text_input("Name", a["name"], key=f"n_{aid}")
            a["enabled"] = st.checkbox("Enabled", a["enabled"], key=f"e_{aid}")
            c1, c2, c3 = st.columns([2, 3, 2])
            a["provider"] = c1.selectbox(
                "Provider", list(PROVIDERS),
                index=list(PROVIDERS).index(a["provider"]), key=f"p_{aid}",
            )
            a["model"] = c2.text_input("Model ID", a["model"], key=f"m_{aid}")
            a["temperature"] = c3.slider("Temperature", 0.0, 1.5, float(a["temperature"]), 0.1, key=f"t_{aid}")
            a["system_prompt"] = st.text_area("System prompt", a["system_prompt"], height=140, key=f"s_{aid}")
            if st.button("🗑️ Delete agent", key=f"d_{aid}"):
                st.session_state.agents = [x for x in st.session_state.agents if x["id"] != aid]
                st.session_state.chats.pop(aid, None)
                st.rerun()

    st.divider()
    st.subheader("➕ Add a position")
    with st.form("add_agent", clear_on_submit=True):
        n = st.text_input("Name", placeholder="e.g. Designer, Tester, Translator")
        p = st.selectbox("Provider", list(PROVIDERS))
        sp = st.text_area("System prompt", placeholder="You are the … Your job is to …")
        if st.form_submit_button("Add agent") and n.strip():
            st.session_state.agents.append(
                new_agent(n.strip(), sp.strip() or f"You are the {n.strip()} of an AI crew.", p)
            )
            st.rerun()

    st.divider()
    st.subheader("Save / load crew")
    st.caption("Streamlit Cloud storage is temporary, so export your crew to keep it.")
    st.download_button(
        "⬇️ Export crew (JSON)",
        json.dumps(st.session_state.agents, indent=2),
        file_name="crew.json",
        mime="application/json",
    )
    up = st.file_uploader("Import crew (JSON)", type="json")
    if up is not None and st.button("Load imported crew"):
        try:
            data = json.loads(up.getvalue())
            loaded = []
            for d in data:
                prov = d.get("provider") if d.get("provider") in PROVIDERS else "Groq"
                loaded.append(
                    new_agent(
                        d.get("name", "Agent"),
                        d.get("system_prompt", ""),
                        prov,
                        d.get("model"),
                        float(d.get("temperature", 0.4)),
                        bool(d.get("enabled", True)),
                    )
                )
            if loaded:
                st.session_state.agents = loaded
                st.session_state.chats = {}
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load crew: {e}")
