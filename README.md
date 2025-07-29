# Trip Agent: Multi-Agent Trip Planner

Trip Agent is a comprehensive trip planning system built using a multi-agent architecture with LangChain, LangGraph, LangSmith, and Pydantic. The system leverages specialized AI agents that collaborate to create personalized travel itineraries based on user preferences, destination information, local events, and logistical constraints.

## Features

- **Conversational Interface**: Chat with the system to plan your trip
- **Interactive Map**: View your trip itinerary on an interactive map
- **Multi-Agent Architecture**: Specialized agents work together to create optimal trip plans
- **Personalized Recommendations**: Tailored suggestions based on your preferences and interests
- **Detailed Itineraries**: Day-by-day schedules with attractions, travel times, and activities

## Architecture

The system is built using a multi-agent architecture with five specialized agents:

1. **User Interface Agent**: Manages user communication and preference capture
2. **Destination Information Agent**: Researches and provides evergreen destination information
3. **Local Events & Logistics Agent**: Handles time-sensitive and logistical information
4. **Itinerary Optimization Agent**: Builds optimized travel schedules
5. **Orchestrator Agent**: Coordinates agent activities and compiles final output

These agents collaborate through a LangGraph workflow to create comprehensive trip plans.

## Project Structure

```
trip-agent/
├── src/
│   ├── agents/             # Agent implementations
│   ├── api/                # FastAPI application
│   ├── memory/             # Memory systems
│   ├── models/             # Pydantic models
│   ├── ui/                 # Streamlit UI
│   └── workflows/          # LangGraph workflows
├── tests/                  # Unit tests
├── .env.example            # Environment variables template
├── main.py                 # Main application entry point
├── PLANNING.md             # Project planning document
├── pyproject.toml          # Poetry configuration and dependencies
├── README.md               # Project documentation
└── TASKS.md                # Project tasks
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trip-agent.git
   cd trip-agent
   ```

2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

5. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   # Add other API keys as needed
   ```

## Usage

Run the application using Poetry:

```bash
poetry run start
```

This will start both the API server and the Streamlit UI. You can then access the UI at http://localhost:8501.

You can also run just the API or UI:

```bash
# Run just the API
poetry run python main.py --mode api

# Run just the UI
poetry run python main.py --mode ui
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangSmith](https://smith.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Folium](https://python-visualization.github.io/folium/)