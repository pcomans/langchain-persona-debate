# AI Roundtable Discussion

Break out of your thought bubble by exploring questions through multiple AI-generated perspectives.

## What It Does

AI Roundtable creates dynamic discussions where AI characters with distinct viewpoints debate any topic you provide. The system:

1. Deploys multiple AI personas with contrasting perspectives
2. Facilitates a structured debate through a moderator
3. Synthesizes key insights from the entire conversation

## Core Features

- Diverse AI characters (Analyst, Visionary, Pragmatist, etc.) with distinct reasoning approaches
- Moderator that selects characters to create productive tension
- Automatic synthesis extracting key insights and areas of agreement/disagreement
- Configurable discussion depth
- Complete discussion export to JSON and markdown

## Technical Requirements

- Python 3.8+
- Google Vertex AI access
- Poetry for dependency management

## Quick Setup

Follow the instructions at https://docs.mindmac.app/how-to.../add-api-key/create-google-cloud-vertex-ai-api-key to generate your API key.

```bash
# Clone and install dependencies
git clone https://github.com/pcomans/langchain-persona-debate
cd langchain-persona-debate
poetry install --no-root

# Set up environment
cp example.env .env
# Edit .env to set GOOGLE_APPLICATION_CREDENTIALS pointing to your Google Cloud key file
# Example: GOOGLE_APPLICATION_CREDENTIALS=.keys/your-key-file.json
```

## Run a Discussion

```bash
# Standard discussion (5 turns)
poetry run python discussion.py -q "Should AI systems require human oversight?"

# Deeper exploration (8 turns)
poetry run python discussion.py -q "How might quantum computing change cryptography?" -t 8

# Shorter discussion (2 turns)
poetry run python discussion.py -q "How can AI improve healthcare?" -t 2
```

## System Architecture

The project operates through:

1. YAML-based configuration files for characters and prompts
2. Dynamic character selection based on conversational context
3. Orchestrated API calls to Vertex AI
4. Structured output generation

## Customization

Modify the system by editing:
- `config/characters.yaml` - Define character personas and their guidance
- `config/prompts.yaml` - Adjust moderator and synthesizer behavior

## Output

Each discussion generates a timestamped directory in the `outputs` folder (e.g., `outputs/roundtable_discussion_YYYYMMDD_HHMMSS/`) containing:
- Individual JSON response files for each character's contribution
- Moderator introduction and transitions
- Complete markdown transcript (once the discussion is complete)
- Synthesized insights summary (once the discussion is complete)

The discussion runs progressively, with each character taking turns to contribute. You can monitor the progress by checking the output directory as files are generated.

## Understanding the Output

The output files are created in real-time as the discussion progresses:
- `question.json.json` - The original question
- `discussion_info.json.json` - Basic metadata about the discussion
- `moderator_introduction.json.json` - The moderator's opening remarks
- `response_[Character_Name].json.json` - Each character's contribution
- `moderator_transition_[N].json.json` - Transitions between characters
- `synthesis.json.json` - Final summary (created at the end)
- `discussion_transcript.md` - Complete discussion in markdown format (created at the end)

## Why Use AI Roundtable

This tool helps you:
- Explore complex questions from multiple angles
- Identify blind spots in your thinking
- Generate comprehensive analysis on any topic
- Extract structured insights from conversational data
