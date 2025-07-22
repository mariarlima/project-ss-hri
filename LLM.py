from openai import OpenAI
import configuration
import os

class LLM:
    def __init__(self, api_key, llm_prompt=None):
        """
        Initialize LLM class
        """
        # Set up OpenAI client using the provided API key
        self.openai_model = OpenAI(api_key=api_key)
        # Use "gpt-4o" as model_id

        # Assign the llm_prompt to a class variable to maintain the conversation context across turns
        self.conversation = []
        self.llm_prompt = llm_prompt

        # Auto-evaluation criteria for elderly-friendly responses
        self.evaluation_criteria = {
            'max_length': 200,  # Maximum characters for elderly-friendly responses
            'forbidden_words': ['death', 'dying', 'kill', 'suicide', 'depressed'],  # Sensitive words to avoid
            'required_tone': ['friendly', 'encouraging', 'patient'],  # Expected tone indicators
        }

    def evaluate_response(self, response):
        """
        Evaluate if the response is appropriate for elderly users
        
        Parameters:
        - response: The generated response to evaluate
        
        Returns:
        - tuple: (is_appropriate: bool, reason: str)
        """
        # Check response length - elderly users prefer shorter responses
        if len(response) > self.evaluation_criteria['max_length']:
            return False, "Response too long"
        
        # Check for forbidden/sensitive words that might upset elderly users
        response_lower = response.lower()
        for word in self.evaluation_criteria['forbidden_words']:
            if word in response_lower:
                return False, f"Contains sensitive word: {word}"
        
        # Check if response is empty or too short
        if len(response.strip()) < 10:
            return False, "Response too short or empty"
        
        # Check for overly complex language (simple heuristic: avg word length)
        words = response.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 7:  # Elderly-friendly responses should use simpler words
                return False, "Language too complex"
        
        return True, "Response appropriate"    

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
        # Add the user message to the conversation history
        self.conversation.append({"role": role, "content": text})

        # If this is the first message in the conversation, prepend the llm_prompt (system prompt)
        if self.llm_prompt and not any(msg["role"] == "system" for msg in self.conversation):
            self.conversation = self.llm_prompt + self.conversation

        # Optionally add an additional system message
        if addition_system_message:
            self.conversation.append({"role": "system", "content": addition_system_message})

        # Call the OpenAI API to get a response using the current conversation context
        response = self.openai_model.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation
        )

        # Extract the assistant's reply
        reply_content = response.choices[0].message.content

        # Know we need to check if the message is appropriate
        evaluation_result = self.evaluate_response(response=reply_content)

        if evaluation_result[0]:

            # Add the assistant's reply to the conversation history
            self.conversation.append({"role": "assistant", "name": "Blossom", "content": reply_content})

            # Return just the content of the response
            return reply_content
        else:
            return "Thank you for sharing that with me."


        
