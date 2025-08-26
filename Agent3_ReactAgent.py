# ReAct Agent - Reasoning and Acting Agent

# Obectives:
# 1. Learn how to create Tools in Langgraph
# 2. How to create a ReAct Graph
# 3. Work with different types of Messages such as ToolMessages
# 4. Test out robustness of our graph.

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# ------------------------------------------------------
# Annotated - Provides additional context without affecting the type itself

# email = Annotated[str, "This has to be a valid email format!"]
# print(email.__metadata__)
# -----------------------------------------------------

# from langgraph.graph.message import add_messages
# add_messages - is "Reducer Function"
# Rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge data into the current state.

# without a Reducer, updates would have replaced the existing value entirely!

# ---------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """multiplication function"""
    return a * b


tools = [add, subtract, multiply]

model = ChatGroq(model="llama3-8b-8192").bind_tools(tools)

def model_call(state:AgentState)-> AgentState:
    system_prompt = SystemMessage(content=
                                "You are my AI Assistant, please answer my query to the best of your ability."
                                )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40+12 and the multiply the result by 6 and subtract by 1. Also tell me a joke")]}
print_stream(app.stream(inputs, stream_mode="values"))