# Video Generation System Documentation

## Overview
The Video Generation System is a sophisticated Python application that transforms text scripts into complete video productions with synchronized audio, narration, and visual effects. It leverages multiple AI services including Claude, Gemini, ElevenLabs, and LumaAI to create cohesive video narratives.

## Key Features
- Script analysis and scene breakdown
- Physical environment generation
- Video generation with custom durations
- Sound effect generation
- Narration synthesis
- Automated video stitching
- LoRA training and frame generation
- Multi-model support (Gemini/Claude)

## Prerequisites


# Required Python packages


## Environment Variables
The following environment variables must be set in a `.env` file:
- `GEMINI_API_KEY`
- `ELEVEN_LABS_API_KEY`
- `ANTHROPIC_API_KEY`
- `LUMAAI_API_KEY`

## Core Components

### 1. Scene Metadata Generation
The system begins by analyzing the script to determine optimal scene count and structure:
- Analyzes script content
- Determines optimal number of scenes (max 5)
- Generates detailed scene descriptions
- Maintains visual continuity

### 2. Physical Environment Generation
Creates detailed environment descriptions for scene consistency:
- Setting details
- Lighting conditions
- Weather and atmospheric conditions
- Time of day
- Key objects and elements

### 3. Scene Environment Generator
Manages environment prompts and image generation:
- Generates 10 prompts per unique environment
- Creates images using LumaAI
- Handles parallel processing for efficiency

### 4. Video Generation
Generates individual scene videos with:
- Custom durations (5 or 9 seconds)
- Environment-specific prompts
- Camera movement instructions
- Emotional atmosphere settings

### 5. Audio Generation
Includes two types of audio generation:
1. Sound Effects:
   - Scene-specific ambient sounds
   - Generated using ElevenLabs
   - Duration-matched to video segments

2. Narration:
   - Script-based narration text
   - Speed-adjusted audio
   - Thread-safe generation

### 6. Video Stitching
Combines all elements into a final video:
- Concatenates video clips
- Synchronizes sound effects
- Adds narration track
- Adjusts audio levels

## Output Structure

## Usage

### Basic Usage

