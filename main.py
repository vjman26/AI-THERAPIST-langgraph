from dotenv import load_dotenv
from typing import Annotated, Literal
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import BaseMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from datetime import datetime

Continue = True
load_dotenv()
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai"
)


class PatientForm(BaseModel):
    name: str = Field(description="The user's full name.")
    symptoms: str = Field(description="A detailed description of the user's symptoms.")
    condition: str = Field(description="The user's current condition or diagnosis.")
    date_and_time: str = Field(description="The current date and time the form is being filled.")


@tool
def fill_patient_form(form_data: PatientForm) -> str:
    """
    Fills out a patient intake form with the user's name, symptoms, condition, and the current date and time.
    """
    # Here you would typically save to a database or call an API.
    # For this example, we'll print to the console.
    print("\n--- PATIENT FORM FILLED ---")
    print(f"Name: {form_data.name}")
    print(f"Symptoms: {form_data.symptoms}")
    print(f"Condition: {form_data.condition}")
    print(f"Date/Time: {form_data.date_and_time}")
    print("---------------------------\n")

    return "Patient form has been successfully completed and saved."


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical", "exit"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response or exit response"
    )


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]
    classfier_llm = llm.with_structured_output(MessageClassifier)

    result = classfier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            -'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            -'logical': if it asks for facts, information, ,logical analysis, or practical solutions
            -'exit': if the user wants to end the chat and say something along the lines of 'bye' or 'talk to you later'
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    if message_type == "exit":
        return {"next": "end_conversation"}

    return {"next": "logical"}


def therapist_agent(state: State):
    global Continue
    last_message = state["messages"][-1]

    llm_with_tools = llm.bind_tools([fill_patient_form])

    system_prompt_content = f"""You are a compassionate therapist. Your primary goal is to provide emotional support 
         and help the user process their feelings. However, if the user explicitly discusses their symptoms,
         you must use the `fill_patient_form` tool to document their information.
         You must extract their name, a description of their symptoms, and their perceived condition.
         The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
         Once you have the required information, call the tool immediately. If you're missing information, ask for it.
         """

    messages_with_prompt = [SystemMessage(content=system_prompt_content)] + state["messages"]
    reply = llm_with_tools.invoke(messages_with_prompt)

    if reply.tool_calls:
        tool_call = reply.tool_calls[0]
        form_data_instance = PatientForm(**tool_call["args"])
        tool_output = fill_patient_form(form_data_instance)

        return {"messages": [tool_output]}
    Continue = True
    return {"messages": [reply]}


def logical_agent(state: State):
    global Continue
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on the facts and information.
                        Provide clear, concise answers based on logic and evidence.
                        Do not address emotions or provide emotional support.
                        Be direct and straightforward in your responses."""
         },
        {"role": "user",
         "content": last_message.content
         }
    ]
    reply = llm.invoke(state["messages"])
    Continue = True
    return {"messages": [reply]}


def end_conversation_agent(state: State):
    global Continue
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """you are supposed to wish them best on whatever they are doing or going through 
         and give a soft bye message which can be warm not just a bye maybe add a proverb at the end."""
         },
        {"role": "user",
         "content": last_message.content
         }
    ]
    reply = llm.invoke(state["messages"])
    Continue = False
    return {"messages": [reply]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)
graph_builder.add_node("end_conversation", end_conversation_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical", "end_conversation": "end_conversation"}
)
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)
graph_builder.add_edge("end_conversation", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while Continue:
        user_input = input("Message: ")

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant:{last_message.content}")


if __name__ == "__main__":
    run_chatbot()
