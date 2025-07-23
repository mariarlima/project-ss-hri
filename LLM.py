from openai import OpenAI
import threading
import time
import configuration
import os
import base64

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

    def prepare_conversation(self, text, role="user", image_path=None, addition_system_message=None):
        """
        Prepare conversation without making API call - for faster processing
        """
        # Add the user message to the conversation history
        self.conversation.append({"role": role, "content": text})

        # If this is the first message in the conversation, prepend the llm_prompt (system prompt)
        if self.llm_prompt and not any(msg["role"] == "system" for msg in self.conversation):
            self.conversation = self.llm_prompt + self.conversation

        # Optionally add an additional system message
        if addition_system_message:
            self.conversation.append({"role": "system", "content": addition_system_message})

        if image_path and os.path.exists(image_path):
            # Handle image input
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                # Create message with both text and image
                message_content = [
                    {"type": "text", "text": text}
                ]
                
                # Add image to the message
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
                
                # Update the last user message with image
                self.conversation[-1] = {
                    "role": role, 
                    "content": message_content
                }
            except Exception as e:
                print(f"Error processing image: {e}")

    def request_response(self, text, role="user", image_path=None, addition_system_message=None):
        """
        Request response with current conversation context - optimized for speed

        Parameters:
        - text: user input
        - role: "user" or "system"
        - image_path: None or an image path
        - addition_system_message: optional extra instruction to guide LLM behaviour

        Returns:
        - The content string from the LLM
        """
        
        # Prepare conversation first (fast operation)
        self.prepare_conversation(text, role, image_path, addition_system_message)

        # Start timing
        start_time = time.time()
        
        try:
            # Call the OpenAI API with optimized parameters for speed
            response = self.openai_model.chat.completions.create(
                model="gpt-4o-mini",  # Faster model
                messages=self.conversation,
                max_tokens=150,  # Limit tokens for faster response
                temperature=0.7,  # Lower temperature for more focused responses
                stream=False  # Could enable streaming for even faster perceived response
            )

            # Extract the assistant's reply
            reply_content = response.choices[0].message.content
            
            print(f"LLM response time: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "I'm having trouble processing that right now. Could you try again?"

        # Quick evaluation check
        evaluation_result = self.evaluate_response(response=reply_content)

        if evaluation_result[0]:
            # Add the assistant's reply to the conversation history
            self.conversation.append({"role": "assistant", "name": "Blossom", "content": reply_content})
            return reply_content
        else:
            # Add fallback response to conversation
            fallback = "Thank you for sharing that with me."
            self.conversation.append({"role": "assistant", "name": "Blossom", "content": fallback})
            return fallback

    def request_response_async(self, text, role="user", image_path=None, addition_system_message=None, callback=None):
        """
        Async version that doesn't block - calls callback when done
        """
        def async_request():
            result = self.request_response(text, role, image_path, addition_system_message)
            if callback:
                callback(result)
        
        threading.Thread(target=async_request, daemon=True).start()

    def trim_conversation(self, max_messages=10):
        """
        Keep conversation history short for faster processing
        """
        if len(self.conversation) > max_messages:
            # Keep system messages and recent messages
            system_msgs = [msg for msg in self.conversation if msg["role"] == "system"]
            recent_msgs = self.conversation[-(max_messages-len(system_msgs)):]
            self.conversation = system_msgs + recent_msgs