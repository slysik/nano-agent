#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from utils.constants import ensure_session_log_dir

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def get_tts_script_path():
    """
    Determine which TTS script to use based on available API keys.
    Priority order: ElevenLabs > OpenAI > pyttsx3
    """
    # Get current script directory and construct utils/tts path
    script_dir = Path(__file__).parent
    tts_dir = script_dir / "utils" / "tts"
    
    # Check for ElevenLabs API key (highest priority)
    if os.getenv('ELEVENLABS_API_KEY'):
        elevenlabs_script = tts_dir / "elevenlabs_tts.py"
        if elevenlabs_script.exists():
            return str(elevenlabs_script)
    
    # Check for OpenAI API key (second priority)
    if os.getenv('OPENAI_API_KEY'):
        openai_script = tts_dir / "openai_tts.py"
        if openai_script.exists():
            return str(openai_script)
    
    # Fall back to pyttsx3 (no API key required)
    pyttsx3_script = tts_dir / "pyttsx3_tts.py"
    if pyttsx3_script.exists():
        return str(pyttsx3_script)
    
    return None


def announce_subagent_completion(custom_message=None):
    """Announce subagent completion using the best available TTS service."""
    try:
        tts_script = get_tts_script_path()
        if not tts_script:
            return  # No TTS scripts available
        
        # Use custom message if provided, otherwise use default
        completion_message = custom_message or "Subagent Complete Steve"
        
        # Call the TTS script with the completion message
        subprocess.run([
            "uv", "run", tts_script, completion_message
        ], 
        capture_output=True,  # Suppress output
        timeout=10  # 10-second timeout
        )
        
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Fail silently if TTS encounters issues
        pass
    except Exception:
        # Fail silently for any other errors
        pass


def check_for_winner_announcement(input_data, log_dir):
    """Check if this is a perf evaluation completion and announce winner if found."""
    try:
        # Check if the subagent name indicates a perf evaluation
        subagent_name = input_data.get("subagent_name", "")
        
        # Look for evaluation results in the transcript
        if "perf" in subagent_name.lower() or "eval" in subagent_name.lower():
            transcript_path = input_data.get("transcript_path", "")
            if os.path.exists(transcript_path):
                # Read the transcript to find winner information
                with open(transcript_path, 'r') as f:
                    lines = f.readlines()
                    
                # Look for the final ranking section
                found_ranking = False
                for i, line in enumerate(lines):
                    try:
                        line_data = json.loads(line.strip())
                        if line_data.get("type") == "text":
                            content = line_data.get("text", "")
                            
                            # Check if we've found the Final Ranking section
                            if "### Final Ranking" in content:
                                found_ranking = True
                            
                            # If we're in the ranking section, look for 1st place
                            if found_ranking and "**1st Place**:" in content:
                                # Extract winner info from the line
                                # Format: "1. **1st Place**: model_name (Overall Grade: X)"
                                import re
                                match = re.search(r'\*\*1st Place\*\*:\s*([^(]+)\s*\(Overall Grade:\s*([^)]+)\)', content)
                                if match:
                                    winner_model = match.group(1).strip()
                                    winner_grade = match.group(2).strip()
                                    
                                    # Announce the winner
                                    winner_message = f"First place winner: {winner_model}, Overall grade: {winner_grade}"
                                    announce_subagent_completion(winner_message)
                                    return True
                    except (json.JSONDecodeError, KeyError):
                        continue
    except Exception:
        pass  # Fail silently
    
    return False


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--chat', action='store_true', help='Copy transcript to chat.json')
        parser.add_argument('--notify', action='store_true', help='Enable TTS notifications')
        args = parser.parse_args()
        
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Extract required fields
        session_id = input_data.get("session_id", "")
        stop_hook_active = input_data.get("stop_hook_active", False)

        # Ensure session log directory exists
        log_dir = ensure_session_log_dir(session_id)
        log_path = log_dir / "subagent_stop.json"

        # Read existing log data or initialize empty list
        if log_path.exists():
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    log_data = []
        else:
            log_data = []
        
        # Append new data
        log_data.append(input_data)
        
        # Write back to file with formatting
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Handle --chat switch (same as stop.py)
        if args.chat and 'transcript_path' in input_data:
            transcript_path = input_data['transcript_path']
            if os.path.exists(transcript_path):
                # Read .jsonl file and convert to JSON array
                chat_data = []
                try:
                    with open(transcript_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    chat_data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass  # Skip invalid lines
                    
                    # Write to logs/chat.json
                    chat_file = os.path.join(log_dir, 'chat.json')
                    with open(chat_file, 'w') as f:
                        json.dump(chat_data, f, indent=2)
                except Exception:
                    pass  # Fail silently

        # Check for winner announcement first
        winner_announced = False
        if args.notify:
            winner_announced = check_for_winner_announcement(input_data, log_dir)
        
        # Announce regular subagent completion if no winner was announced
        if args.notify and not winner_announced:
            announce_subagent_completion()

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == "__main__":
    main()