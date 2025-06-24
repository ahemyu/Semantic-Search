from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from search import SemanticSearch


class LLMSearchAgent:
    def __init__(
        self,
        searcher: SemanticSearch,
        llm_provider: str,
        model_name: str,
        api_key: str,
    ):
        """Initialize the LLM search agent with a specific LLM provider and model."""
        self.searcher = searcher
        self.llm = self._initialize_llm(llm_provider, model_name, api_key)
        self.agent_executor = self._create_agent()

    def _initialize_llm(self, llm_provider: str, model_name: str, api_key: str):
        """Initializes and returns the specified LLM client."""
        provider = llm_provider.lower()
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                max_output_tokens=4000,
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=model_name, openai_api_key=api_key, max_tokens=4000
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model_name, anthropic_api_key=api_key, max_tokens=4096
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def _semantic_search_tool(self, query: str) -> str:
        """Tool function that performs semantic search and returns formatted results."""
        try:
            search_query = query.strip()
            limit = 5
            results = self.searcher.search(search_query, limit=limit)
            if not results:
                return "No products found matching your query."
            formatted_results = [
                f"Product {i+1}:\n{str(result)}"
                for i, result in enumerate(results)
            ]
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with semantic search tool and chat history support."""
        search_tool = Tool(
            name="semantic_search",
            func=self._semantic_search_tool,
            description=(
                "Search for electronic products based on semantic similarity. "
                "Input should be a search query describing what kind of product you're looking for. "
                "The tool will return relevant products with their details."
            ),
        )
        tools = [search_tool]


        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful shopping assistant for electronic products. "
                    "Use the semantic_search tool to find relevant items based on the user's request. "
                    "Provide informative responses and ALWAYS suggest at least one alternative if anything remotely close is available. "
                    "Use the conversation history to understand context for follow-up questions."
                    "If you can answer the query of the user based on the conversation history alone, prioritize that over using the search tool.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm=self.llm, tools=tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    def search_and_respond(
        self, user_query: str, chat_history: list[list[str]]
    ) -> str:
        """Process user query with history and return LLM-generated response."""
        try:
            # The agent executor expects history in a specific format
            processed_history = [
                (msg_type, content)
                for user_msg, ai_msg in chat_history
                for msg_type, content in [("human", user_msg), ("ai", ai_msg)]
            ]

            response = self.agent_executor.invoke(
                {"input": user_query, "chat_history": processed_history}
            )
            return response["output"]
        except Exception as e:
            return f"I encountered an error: {str(e)}"