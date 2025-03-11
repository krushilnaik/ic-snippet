import json
from pprint import pprint

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from pydantic import SecretStr
from typing_extensions import Annotated, Literal, TypedDict


class InputState(TypedDict):
    request: str


class OutputState(TypedDict):
    response: str


class Queries(TypedDict):
    queries: list[str]


class Requirements(TypedDict):
    topic: str
    task: Literal["research", "outline", "drafting", "risk_review"]
    audience: str
    content_type: Literal["email", "blog"]


class State(InputState, OutputState, Requirements, Queries):
    current_step: str
    messages: Annotated[list[BaseMessage], add_messages]
    research: list[str]
    outline: str


model = ChatOpenAI(
    model="llama-3.2-1b-instruct",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("not-needed"),
)


def gather_requirements(state: State):
    prompt = (
        "You are a requirement gatherer. You will read text written by a human and identify the following components of the text: \n"
        "1. The task: this will be either research (user is asking for research gathering), outline (user is asking for an outline), or drafting (user is asking for a first draft).\n"
        "2. The audience: this will be a C-suite title mentioned in the text (CEO, CFO, CMO, etc.). \n"
        "3. The content type: this will be either email, blog post, or executive summary.\n\n"
        "4. The topic: this will be the topic of the content\n\n"
        "Respond in JSON with the above properties.\n\n"
    )

    messages = [SystemMessage(content=prompt), HumanMessage(content=state["request"])]

    requirements = model.with_structured_output(Requirements).invoke(messages)

    print(f"{requirements=}")

    return {"messages": messages, **requirements, "current_step": "research"}


def research_agent(state: State):
    print("generating search queries")

    prompt = (
        "You are a researcher at PwC tasked with providing information about similar companies' views on a user provided topic. "
        "Generate a list of search queries to gather relevant information. Only generate 3 queries max."
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Deloitte, EY, and KPMG's views on {state['topic']}"),
    ]

    # queries = model.with_structured_output(Queries).invoke(messages)

    print("researching ideas")

    content = []
    queries = []

    with open("example_search.json", "r") as file:
        results = json.loads(file.read())

        for result in results:
            queries.append(result["query"])
            content += [r["content"] for r in result["results"]]

    return {"messages": messages, "queries": queries, "research": content}


def outline_agent(state: State):
    prompt = (
        f"Write an outline for a {state['content_type']} about {state['topic'].lower()}. Use the following research as reference. "
        "The research: \n\n"
        "\n\n".join(state["research"])
    )

    messages = [
        HumanMessage(content=prompt),
    ]

    print("writing outline")
    response = model.invoke(messages)

    outline = response.content

    return {"outline": outline}


# checkpointer = MemorySaver()
workflow = StateGraph(input=InputState, output=State, state_schema=State)

workflow.add_node("gather_requirements", gather_requirements)
workflow.add_node("research_agent", research_agent)
workflow.add_node("outline_agent", outline_agent)

workflow.add_edge(START, "gather_requirements")
workflow.add_edge("gather_requirements", "research_agent")
workflow.add_edge("research_agent", "outline_agent")
workflow.add_edge("outline_agent", END)

workflow.set_entry_point("gather_requirements")

graph = workflow.compile()
