import json
import re
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Load Environment Variables and Configuration
# -----------------------------------------------------------------------------
load_dotenv()

# Configuration
GEMINI_MODEL = "gemini-2.0-pro-exp-02-05"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def load_characters(yaml_path: str = "config/characters.yaml") -> Dict[str, Dict[str, str]]:
    """Load character information from a YAML file.
    
    If the file doesn't exist, returns the default hardcoded character data.
    """
    try:
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load characters from {yaml_path}: {e}")
        print("Using default hardcoded character data instead.")
        # Hardcoded values should be defined here, but since we're removing them,
        # this would just raise an exception if the file doesn't exist
        raise

def load_prompts(yaml_path: str = "config/prompts.yaml") -> Dict[str, str]:
    """Load prompts from a YAML file.
    
    If the file doesn't exist, returns the default hardcoded prompts.
    """
    try:
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load prompts from {yaml_path}: {e}")
        print("Using default hardcoded prompts instead.")
        # Hardcoded values should be defined here, but since we're removing them,
        # this would just raise an exception if the file doesn't exist
        raise

def save_output(output_dir: Path, filename: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save output to a file with metadata in a subdirectory."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create the output data with timestamp and content
    output_data = {
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        output_data.update(metadata)
    
    # Save to file
    with open(output_dir / f"{filename}.json", "w") as f:
        json.dump(output_data, f, indent=2)

def save_question(output_dir: Path, question: str) -> None:
    """Save the question to a JSON file."""
    save_output(output_dir, "question.json", question, {"type": "user_question"})

def save_character_response(output_dir: Path, character_name: str, response: str, question: str, model_name: str) -> None:
    """Save a character's response to a JSON file."""
    safe_name = re.sub(r'[^\w\-_\.]', '_', character_name)
    filename = f"response_{safe_name}.json"
    metadata = {
        "character": character_name,
        "question": question,
        "type": "character_analysis",
        "model": model_name
    }
    save_output(output_dir, filename, response, metadata)

def save_synthesis(output_dir: Path, synthesis: str, question: str, model_name: str) -> None:
    """Save the synthesis to a JSON file."""
    safe_model = model_name.replace(".", "_")
    filename = f"synthesis_{safe_model}.json"
    metadata = {
        "question": question,
        "type": "synthesis",
        "model": model_name
    }
    save_output(output_dir, filename, synthesis, metadata)

def _create_character_metadata(question: str, characters: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Create metadata for the conversation."""
    return {
        "question": question,
        "character_count": len(characters)
    }

# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------
class ModeratorOutput(BaseModel):
    """Output model for the moderator to select the next speaker."""
    next_character: str = Field(description="Name of the character who should speak next")
    introduction: str = Field(description="Introduction or transition text for the next speaker")
    reasoning: str = Field(description="Reasoning for why this character was selected to speak next")
    guidance_used: Optional[str] = Field(None, description="Reference to the character guidance that was used in the selection process")

    def __str__(self) -> str:
        """String representation of the moderator's decision."""
        result = f"Next speaker: {self.next_character}\nIntroduction: {self.introduction}\nReasoning: {self.reasoning}"
        if self.guidance_used:
            result += f"\nGuidance Used: {self.guidance_used}"
        return result

# -----------------------------------------------------------------------------
# Roundtable Discussion Class
# -----------------------------------------------------------------------------
class RoundtableDiscussion:
    def __init__(
        self, 
        llm: BaseChatModel,
        synthesis_llm: BaseChatModel,
        output_dir: Path,
        characters: Dict[str, Dict[str, str]],
        moderator_prompt: str,
        synthesizer_prompt: str,
    ):
        self.llm = llm
        self.synthesis_llm = synthesis_llm
        self.characters = characters
        self.moderator_prompt = moderator_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.output_dir = output_dir
        
        # Build chains for each character
        self.character_chains = self._build_character_chains()
        self.synthesis_chain = self._build_synthesis_chain()
        self.moderator_chain = self._build_moderator_chain()
        
        # Maintain discussion history
        self.discussion_history = []
    
    def _build_character_chains(self) -> Dict[str, chain]:
        """Build a chain for each character type."""
        character_chains = {}
        for character_name, character_info in self.characters.items():
            prompt = ChatPromptTemplate.from_messages([
                ("system", character_info["prompt"]),
                ("system", "Previous discussion context:\n{discussion_context}"),
                ("human", "{user_query}"),
            ])
            character_chains[character_name] = prompt | self.llm | StrOutputParser()
        return character_chains
    
    def _build_moderator_chain(self) -> chain:
        """Build a chain for the moderator to guide the discussion.
        
        Returns a chain that produces a ModeratorOutput containing:
        - next_character: The name of the character who should speak next
        - introduction: Text to introduce the next speaker
        - reasoning: Explanation of why this character was chosen
        - guidance_used: Reference to the character guidance that informed the selection
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.moderator_prompt),
            ("system", "The discussion is addressing the question: '{user_query}'"),
            ("system", "Current discussion state:\n{discussion_context}"),
            ("human", "Based on the current discussion, who should speak next and why? When selecting a character, refer to the specific guidance for that character.")
        ])
        return prompt | self.llm.with_structured_output(ModeratorOutput)
    
    def _build_synthesis_chain(self) -> chain:
        """Build a chain for synthesizing the roundtable discussion."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.synthesizer_prompt),
            ("system", "The question was: '{user_query}'. Here is the roundtable discussion to synthesize:\n\n{discussion_context}"),
            ("human", "Create a comprehensive synthesis that weaves together these various perspectives to address the question.")
        ])
        
        return prompt | self.synthesis_llm | StrOutputParser()
    
    def _format_discussion_context(self) -> str:
        """Format the current discussion history for context."""
        if not self.discussion_history:
            return "No previous discussion yet."
        
        formatted = []
        for entry in self.discussion_history:
            if entry["type"] == "character_response":
                formatted.append(f"{entry['character']}:\n{entry['response']}")
            elif entry["type"] == "moderator":
                moderator_text = entry['text']
                if 'guidance_used' in entry and entry['guidance_used']:
                    formatted.append(f"Moderator: {moderator_text}\n(Guidance used: {entry['guidance_used']})")
                else:
                    formatted.append(f"Moderator: {moderator_text}")
        
        return "\n\n".join(formatted)
    
    def conduct_roundtable(self, question: str, num_turns: int = 5) -> Tuple[List[Dict], str]:
        """Conduct a roundtable discussion with multiple characters guided by a moderator.
        
        Args:
            question: The question to discuss
            num_turns: Number of character responses in the discussion
            
        Returns:
            Tuple containing the full discussion history and the final synthesis
        """
        # Save the question
        save_question(self.output_dir, question)
        
        # Start with moderator introduction
        initial_context = {
            "available_characters": "\n".join(f"- {name}" for name in self.characters.keys()),
            "user_query": question,
            "discussion_context": "The discussion is just beginning.",
            "character_guidance": self.characters
        }
        
        # Get initial moderator introduction and first speaker
        moderator_output = self.moderator_chain.invoke(initial_context)
        first_character = moderator_output.next_character
        
        # Get the guidance information if available
        guidance_info = getattr(moderator_output, "guidance_used", None)
        
        # Create the introduction text
        intro_text = f"Welcome to our roundtable discussion. Today we'll be exploring the question: '{question}' I'd like to start with {first_character}."
        
        # Add moderator's introduction to the discussion
        self.discussion_history.append({
            "type": "moderator",
            "text": intro_text,
            "guidance_used": guidance_info
        })
        
        # Save moderator's introduction with guidance information if available
        model_name = getattr(self.llm, "model_name", "unknown")
        intro_metadata = {
            "type": "moderator", 
            "model": model_name
        }
        
        if guidance_info:
            intro_metadata["guidance_used"] = guidance_info
        
        save_output(
            self.output_dir, 
            "moderator_introduction.json", 
            intro_text,
            intro_metadata
        )
        
        current_character = first_character
        
        # Conduct the discussion for specified number of turns
        for turn in range(num_turns):
            print(f"Turn {turn+1}: {current_character} is speaking...")
            
            # Get the character's response
            discussion_context = self._format_discussion_context()
            response = self.character_chains[current_character].invoke({
                "user_query": question,
                "discussion_context": discussion_context
            })
            
            # Add to discussion history
            self.discussion_history.append({
                "type": "character_response",
                "character": current_character,
                "response": response
            })
            
            # Save individual character response
            save_character_response(self.output_dir, current_character, response, question, model_name)
            
            # If not the last turn, get moderator to choose next character
            if turn < num_turns - 1:
                discussion_context = self._format_discussion_context()
                try:
                    moderator_response = self.moderator_chain.invoke({
                        "available_characters": "\n".join(f"- {name}" for name in self.characters.keys()),
                        "user_query": question,
                        "discussion_context": discussion_context,
                        "character_guidance": self.characters
                    })
                    
                    next_character = moderator_response.next_character
                    transition = moderator_response.introduction
                    
                    # Ensure the selected character exists
                    if next_character not in self.characters:
                        print(f"Warning: Moderator selected '{next_character}' which is not a valid character. Choosing a different character.")
                        # Choose a different character than the current one
                        available_characters = list(self.characters.keys())
                        available_characters.remove(current_character) if current_character in available_characters else None
                        next_character = available_characters[0] if available_characters else current_character
                        transition = f"Let's hear from {next_character} now."
                except Exception as e:
                    print(f"Error getting next character from moderator: {e}")
                    # Choose a different character than the current one
                    available_characters = list(self.characters.keys())
                    available_characters.remove(current_character) if current_character in available_characters else None
                    next_character = available_characters[0] if available_characters else current_character
                    transition = f"Let's hear from {next_character} now."
                
                # Add moderator's transition to the discussion
                moderator_text = transition
                # If we have guidance information, add it to the internal record
                guidance_info = getattr(moderator_response, "guidance_used", None)
                
                self.discussion_history.append({
                    "type": "moderator",
                    "text": moderator_text,
                    "guidance_used": guidance_info
                })
                
                # Save moderator's transition with guidance information if available
                transition_metadata = {
                    "type": "moderator", 
                    "model": model_name, 
                    "turn": turn+1
                }
                
                if guidance_info:
                    transition_metadata["guidance_used"] = guidance_info
                
                save_output(
                    self.output_dir, 
                    f"moderator_transition_{turn+1}.json", 
                    transition,
                    transition_metadata
                )
                
                current_character = next_character
        
        # Generate final synthesis
        discussion_context = self._format_discussion_context()
        synthesis = self.synthesis_chain.invoke({
            "discussion_context": discussion_context, 
            "user_query": question
        })
        
        # Save synthesis output
        synthesis_model_name = getattr(self.synthesis_llm, "model_name", "unknown")
        save_synthesis(self.output_dir, synthesis, question, synthesis_model_name)
        
        # Add moderator's conclusion
        self.discussion_history.append({
            "type": "moderator",
            "text": "Thank you all for your insights. Let me summarize what we've discussed today."
        })
        
        # Add synthesis to discussion history
        self.discussion_history.append({
            "type": "synthesis",
            "text": synthesis
        })
        
        # Write markdown output
        self._write_markdown_output(question, synthesis)
        
        return self.discussion_history, synthesis
    
    def _write_markdown_output(self, question: str, synthesis: str) -> Path:
        """Write the discussion transcript to a markdown file."""
        md_path = self.output_dir / "discussion_transcript.md"
        with open(md_path, "w") as md_file:
            md_file.write("# Roundtable Discussion: Multiple Perspectives\n\n")
            md_file.write(f"## Central Question of Discussion\n\n{question}\n\n")
            
            # Add debate introduction
            md_file.write("## Discussion Transcript\n\n")
            
            for entry in self.discussion_history:
                if entry["type"] == "moderator":
                    md_file.write(f"### 🎙️ Moderator\n\n{entry['text']}\n\n")
                elif entry["type"] == "character_response":
                    # Add emoji based on character type to make the debate visually engaging
                    character_emoji = self._get_character_emoji(entry["character"])
                    md_file.write(f"### {character_emoji} {entry['character']}\n\n{entry['response']}\n\n")
                elif entry["type"] == "synthesis":
                    md_file.write(f"## Synthesis of Key Arguments\n\n{entry['text']}\n\n")
        
        return md_path
        
    def _get_character_emoji(self, character_name: str) -> str:
        """Get the emoji for a given character type."""
        if character_name in self.characters and "emoji" in self.characters[character_name]:
            return self.characters[character_name]["emoji"]
        return "💬"  # Fallback emoji for unknown characters
    
    def get_output_dir(self) -> Path:
        return self.output_dir


# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run a roundtable discussion with AI characters"
    )
    parser.add_argument(
        "-q", 
        "--question", 
        type=str, 
        required=True,
        help="The question to discuss"
    )
    parser.add_argument(
        "-t", 
        "--turns", 
        type=int, 
        default=5,
        help="Number of character responses in the discussion (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/roundtable_discussion_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create chat models - use higher temperature to encourage diverse, opinionated responses
    discussion_llm = ChatVertexAI(model=GEMINI_MODEL, temperature=1.2, max_retries=6)
    synthesis_llm = ChatVertexAI(model=GEMINI_MODEL, temperature=1.0, max_retries=6)
    
    # Load character data and prompts from YAML files or fall back to hardcoded data
    characters = load_characters()
    prompts = load_prompts()
    
    # Create metadata for output and save discussion information
    metadata = _create_character_metadata(args.question, characters)
    save_output(output_dir, "discussion_info.json", f"Discussion on: {args.question}", metadata)
    
    # Create and run the discussion system
    discussion_system = RoundtableDiscussion(
        llm=discussion_llm,
        synthesis_llm=synthesis_llm,
        output_dir=output_dir,
        characters=characters,
        moderator_prompt=prompts["moderator"],
        synthesizer_prompt=prompts["synthesizer"]
    )
    
    # Run the discussion with the specified number of turns
    print(f"Running discussion on question: {args.question}")
    print(f"Number of turns: {args.turns}")
    discussion_history, synthesis = discussion_system.conduct_roundtable(
        args.question, 
        num_turns=args.turns
    )
    
    # Print the result
    print("\n\nSYNTHESIS OF DISCUSSION:")
    print(synthesis)
    
    return 0

if __name__ == "__main__":
    main()