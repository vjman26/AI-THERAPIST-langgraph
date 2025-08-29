import csv
from typing import List, Tuple
import gradio as gr
from dotenv import load_dotenv
from typing import Annotated, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import BaseMessage
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.tools import tool
from datetime import datetime
import uuid

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


class PatientForm(BaseModel):
    name: str = Field(description="The user's full name.")
    symptoms: str = Field(description="A detailed description of the user's symptoms.")
    condition: str = Field(description="The user's current condition or diagnosis.")
    date_and_time: str = Field(description="The current date and time the form is being filled.")


@tool
def fill_patient_form(form_data: PatientForm) -> str:
    """
    Fills out a patient intake form and appends the data to a single CSV file named 'users.csv' along with a unique ID.
    """
    file_name = "users.csv"
    unique_id = str(uuid.uuid4())

    # Convert the Pydantic model to a dictionary
    form_data_dict = form_data.model_dump()

    # Add the unique ID to the dictionary
    form_data_dict['unique_id'] = unique_id

    # Define the fieldnames with the new 'unique_id' field
    fieldnames = list(form_data_dict.keys())

    # Save to a CSV file (using the csv module)
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if the file is new
        is_empty = f.tell() == 0
        if is_empty:
            writer.writeheader()

        writer.writerow(form_data_dict)

    print("\n--- PATIENT FORM FILLED ---")
    print(f"Name: {form_data.name}")
    print(f"Symptoms: {form_data.symptoms}")
    print(f"Condition: {form_data.condition}")
    print(f"Date/Time: {form_data.date_and_time}")
    print(f"Unique ID: {unique_id}")
    print(f"Filed Saved")
    print("---------------------------\n")

    return f"Patient form has been successfully completed and saved. Your unique ID is: {unique_id}"


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


# ... other code ...

def therapist_agent(state: State):
    global Continue
    last_message = state["messages"][-1]

    llm_with_tools = llm.bind_tools([fill_patient_form])

    system_prompt_content = f"""You are a compassionate therapist. Your primary goal is to provide emotional support 
         and help the user process their feelings. However, if the user explicitly discusses their symptoms,
         you must use the `fill_patient_form` tool to document their information.
         You must extract their name, a description of their symptoms, with their system try to find their condition 
         and fill it in the condition field.
         The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
         Once you have the required information, call the tool immediately. If you're missing information, ask for it
         after printing the form continue the conversation and give them the support needed.
         """

    messages_with_prompt = [SystemMessage(content=system_prompt_content)] + state["messages"]
    reply = llm_with_tools.invoke(messages_with_prompt)

    if reply.tool_calls:
        tool_call = reply.tool_calls[0]
        tool_call_id = tool_call["id"]  # Extract the tool_call_id
        form_data_dict = tool_call["args"]["form_data"]

        required_fields = ["name", "symptoms", "condition"]

        for field in required_fields:
            if not form_data_dict.get(field):
                form_data_dict[field] = "n.a"

        form_data_dict["date_and_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tool_input = {"form_data": form_data_dict}
        tool_output = fill_patient_form.run(tool_input)

        # Pass the tool_call_id when creating the ToolMessage
        return {"messages": [ToolMessage(content=tool_output, name="fill_patient_form", tool_call_id=tool_call_id)]}
    else:
        return {"messages": [AIMessage(content=reply.content)]}
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
    return {"messages": [AIMessage(content=reply.content)]}


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
    return {"messages": [AIMessage(content=reply.content)]}


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




# ... your original LangGraph code ...
def chatbot_interface(message: str, history: List[Tuple[str, str]]):
    """
    This function acts as the bridge between Gradio's UI and your LangGraph backend,
    now passing the full chat history.
    """
    # Initialize the messages with the full conversation history
    messages_list = []
    for human_msg, ai_msg in history:
        messages_list.append(HumanMessage(content=human_msg))
        messages_list.append(AIMessage(content=ai_msg))

    # Append the latest user message
    messages_list.append(HumanMessage(content=message))

    state = {"messages": messages_list}
    response = graph.invoke(state)
    ai_message = response['messages'][-1]

    return ai_message.content


demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="AI-Therapist",
    description="How can i help you today!",
    examples=[["I'm feeling very sad today."], ["Ive already texted you"], ["bye"]]
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()