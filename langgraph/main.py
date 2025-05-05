# Import necessary libraries and modules
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.graph.message import (
    add_messages,
)
from langchain.chat_models import (
    init_chat_model,
)
from pydantic import (
    BaseModel,
    Field,
)
from typing_extensions import TypedDict


load_dotenv()

# Initialize the language model (LLM) with a specific configuration
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


class MessageClassifier(BaseModel):
    """
    A Pydantic model for classifying messages.

    Attributes:
        message_type (Literal["emotional", "logical"]): Specifies whether the message
        requires an emotional (therapist) or logical response.
    """

    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response.",
    )


class State(TypedDict):
    """
    A TypedDict representing the structure of the chatbot's state.

    Attributes:
        messages (list): A list of messages with additional metadata.
        message_type (str | None): The type of the message, initially None.
    """

    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    """
    Classifies the type of the last message in the state.

    Args:
        state (State): The current state of the chatbot.

    Returns:
        dict: A dictionary containing the classified message type.
    """
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke(
        [
            {
                "role": "system",
                "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """,
            },
            {"role": "user", "content": last_message.content},
        ]
    )
    return {"message_type": result.message_type}


def router(state: State):
    """
    Routes the state to the appropriate agent based on the message type.

    Args:
        state (State): The current state of the chatbot.

    Returns:
        dict: A dictionary specifying the next node to transition to.
    """
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}


def therapist_agent(state: State):
    """
    Handles emotional messages using a therapist agent.

    Args:
        state (State): The current state of the chatbot.

    Returns:
        dict: A dictionary containing the assistant's response.
    """
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    """
    Handles logical messages using a logical agent.

    Args:
        state (State): The current state of the chatbot.

    Returns:
        dict: A dictionary containing the assistant's response.
    """
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.""",
        },
        {"role": "user", "content": last_message.content},
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


# Create a state graph to define the chatbot's flow
graph_builder = StateGraph(State)

# Add nodes (functions) to the graph
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# Define the edges (transitions) between nodes
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

# Add conditional edges based on the "next" state
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"},
)

# Define the end points of the graph
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# Compile the graph into a callable object
graph = graph_builder.compile()


def run_chatbot():
    """
    Runs the chatbot in a loop, allowing the user to interact with it.

    The chatbot processes user input, classifies the message, and routes it
    to the appropriate agent for a response.

    Exits the loop when the user types "exit".
    """
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
