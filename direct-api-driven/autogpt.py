from __future__ import annotations

from typing import List, Optional

from langchain.chains.llm import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import (
    BaseChatMessageHistory,
    Document,
)
from langchain.tools.base import BaseTool
from langchain_community.tools.human.tool import HumanInputRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain_experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain_experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain_experimental.pydantic_v1 import ValidationError
from langchain_community.callbacks import get_openai_callback

class AutoGPT:
    """Agent for interacting with AutoGPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        max_iterations: int = 20,
        feedback_tool: Optional[HumanInputRun] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()
        self.max_iterations = max_iterations

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        max_iterations: int = 20,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Initialize chat history with system message
        chat_history = chat_history_memory or ChatMessageHistory()
        system_message = SystemMessage(content=f"""You are {ai_name}, {ai_role}.
Your decisions must always be made independently without seeking user assistance.
Play to your strengths as an LLM and pursue simple strategies with no legal complications.""")
        chat_history.add_message(system_message)
        
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            max_iterations,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history,
        )

    def run(self, goals: List[str]) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        # Interaction Loop
        loop_count = 0
        while loop_count < self.max_iterations:
            # Discontinue if continuous limit is reached
            loop_count += 1
            try:
                with get_openai_callback() as cb:
                    # Add detailed logging of LLM object state
                    print("\n=== LLM Configuration ===")
                    print(f"LLM Type: {type(self.chain.llm)}")
                    try:
                        print(f"Deployment: {self.chain.llm.deployment_name}")
                        print(f"Endpoint: {self.chain.llm.azure_endpoint}")
                    except Exception as e:
                        print(f"Error accessing LLM attributes: {str(e)}")
                    print("=======================\n")
                    
                    # Pass the messages directly without converting to strings
                    assistant_reply = self.chain.run(
                        goals=goals,
                        messages=self.chat_history_memory.messages,  # Keep as Message objects
                        memory=self.memory,  # Keep as retriever object
                        user_input=user_input,
                    )
                    
                    if assistant_reply is None:
                        print("Warning: Received None response from LLM")
                        continue
                        
                # Print Assistant thoughts
                print("Assistant Reply:", assistant_reply)  # noqa: T201
                
                # Add messages to chat history
                self.chat_history_memory.add_message(HumanMessage(content=user_input))
                self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

                # Get command name and arguments
                try:
                    action = self.output_parser.parse(assistant_reply)
                    if action is None:
                        print("Warning: Failed to parse action from reply")
                        continue
                except Exception as e:
                    print(f"Error parsing action: {str(e)}")
                    continue

                tools = {t.name: t for t in self.tools}
                if action.name == FINISH_NAME:
                    try:
                        return action.args["response"]
                    except Exception as e:
                        print("Error when finishing: ", str(e))
                        return "Finished but with error at final step"
                if action.name in tools:
                    tool = tools[action.name]
                    try:
                        observation = tool.run(action.args)
                    except ValidationError as e:
                        observation = (
                            f"Validation Error in args: {str(e)}, args: {action.args}"
                        )
                    except Exception as e:
                        observation = (
                            f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                        )
                    result = f"Command {tool.name} returned: {observation}"
                elif action.name == "ERROR":
                    result = f"Error: {action.args}. "
                else:
                    result = (
                        f"Unknown command '{action.name}'. "
                        f"Please refer to the 'COMMANDS' list for available "
                        f"commands and only respond in the specified JSON format."
                    )

                memory_to_add = (
                    f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
                )
                if self.feedback_tool is not None:
                    feedback = f"\n{self.feedback_tool.run('Input: ')}"
                    if feedback in {"q", "stop"}:
                        print("EXITING")  # noqa: T201
                        return "EXITING"
                    memory_to_add += feedback

                self.memory.add_documents([Document(page_content=memory_to_add)])
                self.chat_history_memory.add_message(AIMessage(content=result))
                
            except Exception as e:
                print("\n=== Error Details ===")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                print("==================\n")
                continue