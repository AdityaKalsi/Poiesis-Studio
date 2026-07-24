# 🎬 Film Agents — Multi-Agent AI Filmmaking Pipeline

A **LangGraph-based multi-agent system** that simulates the **film pre-production workflow** using LLMs. The project focuses on the **reasoning and planning** behind filmmaking, transforming a screenplay into structured filmmaking plans before any image or video generation.

> **Current Scope:** Planning & reasoning only. Video generation will be integrated in future iterations.

## Pipeline

```text
Script
   ↓
Script Analyst
   ↓
Scene Design
   ↓
Cinematography
   ↓
Cinematography Critic
      ↺ (Revision Loop)
   ↓
Editing
   ↓
Final Critic
   ↓
END
```

## Features

- 🎭 Multi-agent workflow built with **LangGraph**
- 📖 Structured screenplay analysis
- 🎥 Automated shot list generation
- 🔄 Critic-driven revision loop for quality improvement
- 📦 Typed communication using **Pydantic** schemas
- 📄 Full pipeline output exported as structured JSON

## Tech Stack

- Python
- LangGraph
- LangChain
- OpenAI / Anthropic LLMs
- Pydantic

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`.

## Run

```bash
python main.py
```

Or run with your own screenplay:

```bash
python main.py your_script.txt
```

The complete pipeline output is saved as **output.json**.

## Future Work

- 🎞️ Storyboard generation
- 🎬 AI video generation
- 🎵 Sound & music planning
- 💡 Lighting and VFX agents
- 👥 Human-in-the-loop review
- ⚡ Parallel scene processing

---

Inspired by research including **MovieAgent**, **CameraArtis**, **MAViS**, and **STAGE**.