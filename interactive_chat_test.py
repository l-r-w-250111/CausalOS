import torch
import os
from CausalOS_v1 import UnifiedCausalOS
from CausalChatAgent import CausalChatAgent

def main():
    print("Initializing CausalOS and Chat Agent...")
    osys = UnifiedCausalOS()
    agent = CausalChatAgent(osys)
    
    print("\n=== CausalOS Physics-Based Chat ===")
    print("Ask 'What if' questions. Type 'exit' to quit.\n")
    
    # Pre-warm or test with a known scenario
    # scenario = "If a plane landed in the forest instead of a bird, what happens?"
    # print(f"User: {scenario}")
    # agent.chat(scenario)
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
