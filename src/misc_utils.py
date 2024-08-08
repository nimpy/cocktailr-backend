def dict_to_conversation(data):
    # Convert history to conversation format
    conversation = "\n".join([
        f"{'Agent' if msg['sender'] == 'agent' else 'User'}: {msg['text']}"
        for msg in data['history']
    ])
    
    # Add the new message
    conversation += f"\nUser: {data['newMessage']}"
    
    return conversation
