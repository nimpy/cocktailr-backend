from langchain_core.messages import HumanMessage, AIMessage

def dict_to_conversation(data):
    # Convert history to conversation format
    conversation = "\n".join([
        f"{'Agent' if msg['sender'] == 'agent' else 'User'}: {msg['text']}"
        for msg in data['history']
    ])
    
    # Add the new message
    conversation += f"\nUser: {data['newMessage']}"
    
    return conversation


def dict_to_messages(input_dict):
    messages = []
    
    # Process history
    for message in input_dict.get("history", []):
        if message["sender"] == "agent":
            messages.append(AIMessage(content=message["text"]))
        elif message["sender"] == "user":
            messages.append(HumanMessage(content=message["text"]))
    
    # Add new message
    new_message = input_dict.get("newMessage")
    if new_message:
        messages.append(HumanMessage(content=new_message))
    
    return {"messages": messages}

