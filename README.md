# LLM Agents Final Project
> AI-Powered Shopping Assistant

Hi!

You will be expected to finish this on your own, but you can use the available channels on Discord to ask questions and help others. Please read the entire README and ASSIGNMENT.md before starting, this will give you a better idea of what you need to accomplish.

## The Business problem

You are working for a major e-commerce company that wants to revolutionize their customer service experience using AI. They have requested the Data Science team to build an intelligent shopping assistant that can help customers find products, manage their shopping cart, and handle support requests automatically.

The company wants to explore two main areas: **Customer Experience** and **Operational Efficiency**.

Basically, they would like to provide customers with a conversational AI that can understand natural language queries about products, help them discover items they might like, manage their shopping cart, and automatically escalate complex issues to human support when needed. On the operational side, they want to reduce the workload on human customer service representatives while maintaining high-quality support.

The system should be able to handle scenarios like customers searching for "healthy breakfast options", adding items to their cart, asking for refunds, or needing help with defective products. It should maintain conversation context across multiple turns and know when to bring in human supervisors for sensitive operations.

## About the data

You will consume and use data from a grocery store dataset containing product information, customer orders, and purchase history.

The dataset includes information about 49,000+ grocery products with names, departments, aisles, and pricing, along with historical order data showing what customers have purchased. The data is organized hierarchically with departments containing aisles, which contain individual products.

The system uses this data in two ways: structured search for SQL-like filtering over the catalog for precise queries, and semantic search using vector embeddings for understanding natural language product requests. Additionally, the system maintains conversation state and shopping cart information in memory during user sessions.

## Technical aspects

The team decided to build a conversational AI system using modern Large Language Model (LLM) technologies combined with traditional data processing techniques. The system needs to handle real-time conversations while maintaining state and context across multiple conversation turns.

The technologies involved are:
- Python as the main programming language
- LangChain for LLM integration and tool management
- LangGraph for conversation flow orchestration
- OpenAI GPT models for natural language understanding
- Chroma vector database for semantic product search
- HuggingFace embeddings for text vectorization
- Pandas for data manipulation and filtering
- Pydantic for data validation and schema definition
- Streamlit for the web interface
- Pytest for comprehensive testing

## Installation

Before starting, you must have:
- Python 3.10+
- An OpenAI API key (Or change for your model of choice!)
- Git (for cloning if using Google Colab)

A `requirements.txt` file is provided with all the needed Python libraries for running this project. For installing the dependencies just run:

```console
$ pip install -r requirements.txt
```

**Important**: Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

*Note:* We encourage you to install those inside a virtual environment.

Please see `ASSIGNMENT.md` for detailed instructions on building the vector database, which is required before the system will work.

## Code Style

Following a style guide keeps the code's aesthetics clean and improves readability, making contributions and code reviews easier. Automated Python code formatters make sure your codebase stays in a consistent style without any manual work on your end. If adhering to a specific style of coding is important to you, employing an automated to do that job is the obvious thing to do. This avoids bike-shedding on nitpicks during code reviews, saving you an enormous amount of time overall.

We use [Black](https://black.readthedocs.io/) for automated code formatting in this project, you can run it with:

```console
$ black --line-length=88 .
```

Wanna read more about Python code style and good practices? Please see:
- [The Hitchhiker's Guide to Python: Code Style](https://docs.python-guide.org/writing/style/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Tests

We provide comprehensive unit tests and integration tests along with the project that you can run and check from your side the code meets the minimum requirements of correctness needed to approve. To run just execute:

```console
$ pytest tests/
```

## Running the Application

Once you have completed the TODO functions and built the vector database, you can test the shopping assistant:

### Command Line Testing
```python
from src.conversation_runner import run_single_turn
result = run_single_turn("Hi, I need some bananas", "test-thread-123")
print(result)
```

### Web Interface
```console
$ streamlit run app.py
```

This will start a web interface where you can chat with the shopping assistant, test different conversation flows, and see the cart management in action.
