from typing import TypedDict, Annotated

from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from inference_auth_token import get_access_token

from huijulee_tools import (
    name_to_smiles_and_xyz,
    mace_single_point_energy,
    mace_geometry_optimization,
)


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State):
    """
    Decide whether we should go to the tools node or move to the reporting agent.

    If the last AI message contains tool_calls, we route to 'tools',
    otherwise we are done with tools and can go to 'report_agent'.
    """
    # `state` should be a dict with a "messages" key managed by LangGraph
    messages = state.get("messages", [])
    if not messages:
        raise ValueError(f"No messages found in input state to route_tools: {state}")

    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return "report_agent"


# ============================================================
# 3. LLM nodes (agents)
# ============================================================
def planner_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = (
        "You are a chemistry planning assistant. "
        "You can call tools to:\n"
        "- Convert molecule names to 3D coordinates (XYZ files)\n"
        "- Run MACE single-point energy calculations\n"
        "- Run MACE geometry optimizations\n\n"
        "Decide which tools to call and in what order to study given molecules. "
        "Do not fabricate results; always use tools for energies or geometries."
    ),
):
    """
    Main agent that plans which tools to use and in what order.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        # We simply show the current conversation state to the LLM
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


def report_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = (
        "You are an assistant that returns ONLY valid JSON.\n\n"
        "You will be given the full conversation, including tool calls and tool results. "
        "Extract from it a clean summary of the calculations that were performed.\n"
        "Return a JSON object with the following top-level keys:\n"
        "- 'molecules': a list of objects with keys 'name', 'smiles', 'natoms', 'files'\n"
        "- 'calculations': a list of calculation results (energies, optimization info, etc.)\n"
        "- 'notes': a short free-text explanation (string)\n\n"
        "Do not include any extra text outside the JSON."
    ),
):
    """
    Second agent that turns the whole conversation into a neat JSON report.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 4. LLM / tools setup (ALCF endpoint)
# ============================================================
# Get token for your ALCF inference endpoint
access_token = get_access_token()

# Initialize the model hosted on the ALCF endpoint
llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    # model_name="Qwen/Qwen3-32B",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Tools that the planner agent can call
tools = [
    name_to_smiles_and_xyz,
    mace_single_point_energy,
    mace_geometry_optimization,
]


# ============================================================
# 5. Build the graph
# ============================================================
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node(
    "planner_agent",
    lambda state: planner_agent(state, llm=llm, tools=tools),
)
graph_builder.add_node(
    "report_agent",
    lambda state: report_agent(state, llm=llm),
)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Edges:
# START -> planner_agent
graph_builder.add_edge(START, "planner_agent")

# After planner_agent, decide whether to call tools or go to reporting
graph_builder.add_conditional_edges(
    "planner_agent",
    route_tools,
    {
        "tools": "tools",
        "report_agent": "report_agent",
    },
)

# After tools run, go back to planner_agent to possibly use the new results
graph_builder.add_edge("tools", "planner_agent")

# After report_agent, we terminate
graph_builder.add_edge("report_agent", END)

# Compile the graph
graph = graph_builder.compile()


# ============================================================
# 6. Run / stream the graph
# ============================================================
if __name__ == "__main__":
    prompt = (
        "For the molecules ethanol and acetone:\n"
        "- Build 3D coordinates\n"
        "- Run MACE single-point energy calculations\n"
        "- Run MACE geometry optimizations\n"
        "Then return a JSON summary of all results."
    )

    for chunk in graph.stream(
        {"messages": prompt},
        stream_mode="values",
    ):
        new_message = chunk["messages"][-1]
        new_message.pretty_print()

