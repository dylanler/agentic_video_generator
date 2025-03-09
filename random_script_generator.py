import random
import os
import json
import argparse
from google.generativeai import GenerativeModel
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API clients
def initialize_llm_clients():
    """Initialize the LLM clients based on available API keys."""
    clients = {}
    
    # Initialize Gemini if API key is available
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        clients["gemini"] = True
    
    # Initialize Claude if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        clients["claude"] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    return clients

def generate_random_elements():
    """Generate random elements for the script."""
    
    # Lists of possible elements
    characters = [
        "a detective", "a robot", "a talking animal", "a superhero", "a ghost",
        "a time traveler", "a wizard", "an alien", "a pirate", "a ninja",
        "a scientist", "a lost child", "a mysterious stranger", "a spy", "a chef",
        "a musician", "an astronaut", "a vampire", "a werewolf", "a zombie",
        "a retired assassin", "a runaway bride", "a shapeshifter", "a bounty hunter", "a psychic",
        "a fallen angel", "a cyborg", "a mermaid", "a dragon tamer", "a witch",
        "a billionaire", "a street urchin", "a guardian of ancient secrets", "a clone", "a demigod",
        "a professional thief", "a cursed immortal", "a dimension hopper", "a fairy", "a samurai",
        "a hacker", "a monster hunter", "an exorcist", "a quantum physicist", "a mythological creature",
        "a former cult member", "a sentient AI", "a dream walker", "a reality TV star", "a shaman",
        "a gladiator", "a paranormal investigator", "a royal heir", "a mad scientist", "a spirit medium",
        "a deep sea explorer", "a rogue android", "a memory collector", "a plague doctor", "a celestial being",
        "a professional gambler", "a time police officer", "a cryptid researcher", "a space pirate", "a necromancer",
        "a reformed villain", "a shadow walker", "a truth seeker", "a dimensional architect", "a soul harvester"
    ]
    
    objects = [
        "a mysterious box", "an ancient artifact", "a magical book", "a lost treasure",
        "a futuristic device", "a cursed necklace", "a secret map", "a powerful weapon",
        "a time machine", "a portal", "a forgotten letter", "a hidden camera",
        "a special key", "a rare plant", "a stolen painting", "a broken watch",
        "a glowing crystal", "a strange potion", "a haunted mirror", "a robotic pet",
        "a sentient sword", "a memory eraser", "a holographic diary", "a reality-bending prism",
        "a soul-capturing lantern", "a weather-controlling orb", "a dream recorder", "a quantum compass",
        "a mind-reading helmet", "a door to nowhere", "a living tattoo", "a mechanical heart",
        "a bottomless bag", "a truth-revealing monocle", "a wish-granting monkey's paw", "a ghost-trapping box",
        "a shrinking potion", "a teleportation stone", "a book that predicts the future", "a memory crystal",
        "a device that stops time", "a mirror to parallel worlds", "a music box that controls emotions", "a map to the afterlife",
        "a communication device to the dead", "a pen that makes drawings come alive", "a ring of invisibility", "a vial of immortality elixir",
        "a puzzle box that opens dimensions", "a mask that reveals true intentions", "a camera that photographs the past", "a seed that grows instant structures",
        "a pair of glasses that see through lies", "a flute that controls animals", "a cloak of shadows", "a device that translates any language",
        "a compass that points to what you desire most", "a book with infinite pages", "a quill that writes reality", "a locket containing forgotten memories",
        "a coin that always returns to its owner", "a snow globe of miniature universes", "a clock that reverses aging", "a feather of a phoenix"
    ]
    
    environments = [
        "a dense forest", "an abandoned mansion", "a futuristic city", "a desert island",
        "a space station", "an underwater kingdom", "a medieval castle", "a post-apocalyptic wasteland",
        "a busy marketplace", "a secret laboratory", "a haunted house", "a magical realm",
        "a snowy mountain", "a volcanic island", "a hidden cave", "an ancient temple",
        "a floating city", "a parallel dimension", "a virtual reality world", "a prehistoric jungle",
        "a cyberpunk metropolis", "a steampunk airship", "a crystalline palace", "a subterranean civilization",
        "a city in the clouds", "a desert oasis", "a neon-lit nightclub district", "a forgotten space colony",
        "a pocket dimension", "a world inside a painting", "a library containing all knowledge", "a sentient forest",
        "a city built on the back of a giant creature", "a realm where physics don't apply", "a labyrinth of endless corridors", "a frozen tundra",
        "a tropical paradise", "an abandoned amusement park", "a world between dreams", "a city inside a massive computer",
        "a dimension where time flows backwards", "a realm of perpetual twilight", "a world made entirely of candy", "a floating archipelago",
        "a city beneath the ocean floor", "a realm of pure energy", "a world inside a snow globe", "a massive space ark",
        "a planet with multiple suns", "a world where music is visible", "a city built from living plants", "a realm of shifting realities",
        "a world inside a bubble", "a dimension of pure thought", "a city in the eye of an eternal storm", "a realm where shadows are alive",
        "a world made of clockwork", "a city that only exists at night", "a realm of infinite staircases", "a world inside a mirror",
        "a dimension where colors have taste", "a city that changes location daily", "a world where gravity is optional", "a realm of living stone"
    ]
    
    atmospheres = [
        "mysterious", "tense", "whimsical", "romantic", "eerie",
        "adventurous", "melancholic", "chaotic", "peaceful", "suspenseful",
        "magical", "dystopian", "nostalgic", "surreal", "comedic",
        "dramatic", "horrifying", "inspiring", "dreamlike", "action-packed",
        "claustrophobic", "ethereal", "foreboding", "jubilant", "serene",
        "oppressive", "liberating", "psychedelic", "contemplative", "frantic",
        "desolate", "vibrant", "haunting", "absurd", "enchanting",
        "gritty", "fantastical", "ominous", "idyllic", "tense",
        "bizarre", "tranquil", "unsettling", "exhilarating", "somber",
        "hypnotic", "disorienting", "euphoric", "menacing", "bittersweet",
        "otherworldly", "primal", "transcendent", "uncanny", "feverish",
        "hallucinatory", "meditative", "nightmarish", "playful", "savage",
        "celestial", "industrial", "primordial", "utopian", "visceral",
        "arcane", "clinical", "folkloric", "grotesque", "harmonious"
    ]
    
    storylines = [
        "a quest to find a lost treasure", "solving a mysterious disappearance",
        "escaping from a dangerous situation", "discovering a hidden truth",
        "learning to use newfound powers", "reconciling with the past",
        "preventing a disaster", "adapting to a new reality",
        "overcoming a personal challenge", "forming an unlikely friendship",
        "a journey of self-discovery", "a race against time",
        "a battle against a powerful enemy", "a transformation or metamorphosis",
        "a rebellion against authority", "a love story across boundaries",
        "a coming-of-age experience", "a moral dilemma",
        "a heist or elaborate plan", "a survival story",
        "a revenge mission", "a redemption arc", "a deal with a supernatural entity",
        "a competition with high stakes", "a journey to another world", "a sacrifice for the greater good",
        "a mystery involving doppelgangers", "a conspiracy unraveling", "a prophecy coming true",
        "a curse that must be broken", "a forgotten memory resurfacing", "a test of loyalty",
        "a forbidden knowledge being revealed", "a power struggle", "a series of impossible events",
        "a search for a missing person", "a battle against inner demons", "a game where the rules keep changing",
        "a mission to save a dying world", "a journey to bring back something lost", "a ritual to restore balance",
        "a negotiation between opposing forces", "a mystery involving parallel timelines", "a quest for immortality",
        "a struggle against fate", "a journey to the afterlife and back", "a battle against an ancient evil",
        "a mission to deliver a crucial message", "a search for belonging", "a conflict between duty and desire",
        "a mystery involving mistaken identity", "a journey to recover lost memories", "a struggle against corruption",
        "a mission to protect a powerful secret", "a quest to fulfill a dying wish", "a battle against an unstoppable force",
        "a journey to find a legendary place", "a mystery involving time loops", "a struggle to maintain humanity",
        "a mission to unite warring factions", "a quest to restore a broken world", "a battle against a spreading darkness",
        "a journey to break an ancient cycle", "a mystery involving body-swapping", "a struggle against extinction"
    ]
    
    artistic_styles = [
        "film noir", "anime", "stop-motion", "documentary", "fantasy",
        "sci-fi", "horror", "western", "cyberpunk", "steampunk",
        "claymation", "hand-drawn animation", "watercolor", "3D animation", "silent film",
        "found footage", "mockumentary", "experimental", "surrealist", "minimalist"
    ]
    
    # Randomly select elements
    selected_elements = {
        "characters": random.sample(characters, k=random.randint(1, 3)),
        "objects": random.sample(objects, k=random.randint(1, 2)),
        "environment": random.choice(environments),
        "atmosphere": random.choice(atmospheres),
        "storyline": random.choice(storylines),
        "artistic_style": random.choice(artistic_styles)
    }
    
    return selected_elements

def create_prompt(elements, model="gemini"):
    """Create a prompt for the LLM based on the random elements."""
    
    characters_str = ", ".join(elements["characters"])
    objects_str = ", ".join(elements["objects"])
    
    # Use the same prompt regardless of model
    prompt = f"""
    Create a short film script (approximately 100 words) with the following elements:
    
    Characters: {characters_str}
    Objects: {objects_str}
    Setting: {elements["environment"]}
    Atmosphere: {elements["atmosphere"]}
    Story: {elements["storyline"]}
    Artistic Style: {elements["artistic_style"]}
    
    The script should be concise but vivid, focusing on visual storytelling.
    Format the script as a single paragraph of continuous prose that describes what happens in the film.
    Do not include dialogue formatting, scene headings, or camera directions.
    Just write a descriptive paragraph that could be used to create a short film.
    It should contain the artistic style of the script.
    """
    
    return prompt.strip()

def generate_script_with_llm(prompt, model="gemini", clients=None):
    """Generate a script using the specified LLM."""
    
    if not clients:
        clients = initialize_llm_clients()
    
    if model == "gemini" and "gemini" in clients:
        gemini_model = GenerativeModel("gemini-2.0-flash-001")
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    
    elif model == "claude" and "claude" in clients:
        claude_client = clients["claude"]
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=500,
            temperature=0.7,
            system="You are a creative scriptwriter who specializes in concise, visual storytelling.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text.strip()
    
    else:
        raise ValueError(f"Model '{model}' is not available or not supported.")

def generate_random_script(model="gemini"):
    """Generate a random script using the specified LLM."""
    
    # Initialize LLM clients
    clients = initialize_llm_clients()
    
    # Check if the requested model is available
    if model not in clients:
        available_models = list(clients.keys())
        if not available_models:
            raise ValueError("No LLM API keys are configured. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY in your .env file.")
        model = available_models[0]
        print(f"Requested model '{model}' is not available. Using '{model}' instead.")
    
    # Generate random elements
    elements = generate_random_elements()
    
    # Create prompt
    prompt = create_prompt(elements, model)
    
    # Generate script
    script = generate_script_with_llm(prompt, model, clients)
    
    # Return both the script and the elements used to generate it
    return {
        "script": script,
        "elements": elements
    }

def save_script_to_file(script_data, output_file=None):
    """Save the generated script to a file."""
    
    if not output_file:
        output_file = "random_script.txt"
    
    # Save the script text
    with open(output_file, "w") as f:
        f.write(script_data["script"])
    
    # Save the elements used to generate the script
    elements_file = os.path.splitext(output_file)[0] + "_elements.json"
    with open(elements_file, "w") as f:
        json.dump(script_data["elements"], f, indent=2)
    
    return output_file, elements_file

def main():
    parser = argparse.ArgumentParser(description="Generate a random short film script")
    parser.add_argument("--model", type=str, choices=["gemini", "claude"], default="gemini",
                        help="LLM to use for script generation (default: gemini)")
    parser.add_argument("--output", type=str, default="random_script.txt",
                        help="Output file for the generated script (default: random_script.txt)")
    parser.add_argument("--video_gen", action="store_true",
                        help="Generate a video using the random script")
    parser.add_argument("--video_engine", type=str, choices=["luma", "ltx"], default="luma",
                        help="Video generation engine to use (default: luma)")
    parser.add_argument("--max_scenes", type=int, default=5,
                        help="Maximum number of scenes to generate (default: 5)")
    parser.add_argument("--max_environments", type=int, default=3,
                        help="Maximum number of environments to use (default: 3)")
    parser.add_argument("--skip_narration", action="store_true",
                        help="Skip narration generation")
    parser.add_argument("--skip_sound_effects", action="store_true",
                        help="Skip sound effects generation")
    
    args = parser.parse_args()
    
    try:
        # Generate random script
        script_data = generate_random_script(args.model)
        
        # Save script to file
        script_file, elements_file = save_script_to_file(script_data, args.output)
        
        print(f"Random script generated and saved to: {script_file}")
        print(f"Script elements saved to: {elements_file}")
        print("\nGenerated Script:")
        print("-" * 80)
        print(script_data["script"])
        print("-" * 80)
        
        # Generate video if requested
        if args.video_gen:
            try:
                import video_generation
                print(f"Generating video using {args.video_engine}...")
                
                # Call the video generation function
                scenes_json, final_video = video_generation.generate_video(
                    script_data["script"],
                    model_choice=args.model,
                    video_engine=args.video_engine,
                    max_scenes=args.max_scenes,
                    max_environments=args.max_environments,
                    skip_narration=args.skip_narration,
                    skip_sound_effects=args.skip_sound_effects
                )
                
                if final_video:
                    print(f"Final video saved to: {final_video}")
                
            except ImportError:
                print("Error: video_generation module not found. Make sure it's in the same directory.")
            except Exception as e:
                print(f"Error generating video: {str(e)}")
        
    except Exception as e:
        print(f"Error generating random script: {str(e)}")

if __name__ == "__main__":
    main() 