from openai import OpenAI
import os

class LLM:
    def __init__(self, api_key, llm_prompt=None):
        """
        Initialize LLM class
        """
        # TODO: Set up OpenAI client using the provided API key
        # TODO: Use "gpt-4o" as model_id
        # TODO: Assign the llm_prompt to a class variable to maintain the conversation context across turns

        # Working example
        if llm_prompt is None:
            llm_prompt = []
        self.openai = OpenAI(api_key=api_key)
        self.model_id = "gpt-4o"
        self.conversation = llm_prompt


    def request_response(self, text, role="user", addition_system_message=None):
        """
        Request response with current conversation context

        Parameters:
        - text: user input
        - role: "user" or "system"
        - addition_system_message: optional extra instruction to guide LLM behaviour

        Returns:
        - The content string from the LLM
        """
        # TODO: Create a response dict for the user input with "role" and "context" keys and append it to self.conversation
        # TODO: Consider the optional addition_system_message
        # TODO: Call the OpenAI API to get a response using the current conversation context
        # TODO: Create a dict from llm_response with {"role": "system", "name": "Blossom", "content": <response_text>} and append to the conversation history
        # TODO: Return just the content of the llm_response

        # Working example
        user_response_to_prompt = {"role": role, "content": text}
        self.conversation.append(user_response_to_prompt)
        if addition_system_message:
            self.conversation.append({"role": "system", "content": addition_system_message})
            
        llm_response = self.openai.chat.completions.create(
            model=self.model_id,
            messages=self.conversation
        )

        llm_response_to_prompt = {
            "role": "system",
            "name": "Blossom",
            "content": llm_response.choices[0].message.content
        }
        self.conversation.append(llm_response_to_prompt)

        return llm_response.choices[0].message.content


        
