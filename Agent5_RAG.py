from dotenv import load_dotenv
import os
import jq
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, SystemMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_core.tools import tool


load_dotenv()

llm = ChatGroq(
    model = "llama3-8b-8192", temperature = 0 # I want minimum hallucination so kept temp - 0
)

# Our embedding model has should compatible with - LLM
embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
data_path = "tirumala_english_cleaned.json"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"JSON file not found: {data_path}")

# Since root is a list, we use jq_schema=".[]"
loader = JSONLoader(
    file_path=data_path,
    jq_schema=".[]",          # iterate over each list item
    text_content=False        # prevents forcing list into string
)

try:
    docs = loader.load()
    print(f"JSON loaded successfully with {len(docs)} document(s)")
    print("Preview:", docs[0].page_content[:300])
except Exception as e:
    print(f"Error loading JSON: {e}")
    raise

# check if pdf is here
try:
    docs = loader.load()
    print(f"JSON loaded successfully with {len(docs)} records")
    print("Example:", docs[0].page_content, docs[0].metadata)
except Exception as e:
    print(f"Error loading PDF:{e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(docs)

# persist_directory = r"H:\Prathik\LangGraph_FreeCodeCamp"
# collection_name = "tirumala_docs"

# If our collection doesnot found in our directory, we create using the os command
# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)

try:
    # here we actually we create the vectorstore database
    vectorstore = FAISS.from_documents(
        documents = pages_split,
        embedding = embeddings
        #persist_directory = persist_directory,
        #collection_name = collection_name
    )
    print(f"Created FAISS vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise



# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5} # K is amount of chunks to return where default is '4'.
)

# -------------------------------------------------------------

# start creating tool
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the tirumala document.
    """
    docs_ = retriever.invoke(query)

    if not docs_:
        return "I found no relevent information in the Tirumala document."
    
    results = []
    for i, doc in enumerate(docs_):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt= """You are an intelligent AI assistant who answers questions about 
Tirumala Darshan, Temple services, Pilgrim facilities, traditions, and real-time guidance. 
Use the retriever tool available to answer questions about Tirumala-related data, 
including darshan timings, seva details, accommodation, route navigation, 
historical/cultural insights, and current announcements.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the llm with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool:{t['name']} with query: {t['args'].get('query', 'No query Provided.')}")

        if not t['name'] in tools_dict: # check if a valid tool is present
            print("\n Tool: {t['name]} does not exist.")
            result = "Incorrect Tool name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'.get('query', '')])
            print(f"Result Length: {len(str(result))}")

        # Append the ToolMessage
        results.append(ToolMessage(tools_call_id=['id'], name = t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the Model.")
    return {'messages': results}



graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")


rag_agent = graph.compile()

def running_agent():
    print("\n ====RAG AGENT ======")

    while True:
        user_input = input("\n What is your Question : ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)] # Convert back to HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        print("\n ===== ANSWER =====")
        print(result['messages'][-1].content)

running_agent()
