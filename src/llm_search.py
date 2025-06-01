from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from search import SemanticSearch


class LLMSearchAgent:
    def __init__(self, searcher: SemanticSearch, api_key: str):
        """Initialize the LLM search agent with semantic search capability."""
        self.searcher = searcher
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            max_output_tokens=4000
        )
        self.agent_executor = self._create_agent()

    def _semantic_search_tool(self, query: str) -> str:
        """Tool function that performs semantic search and returns formatted results."""
        try:
            # Parse the query to extract search term
            search_query = query.strip()
            limit = 5  # default limit
            
            results = self.searcher.search(search_query, limit=limit)
            
            if not results:
                return "No products found matching your query."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                product_info = f"Product {i}:\n"
                product_info += str(result)
                formatted_results.append(product_info)
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with semantic search tool."""
        # Define the semantic search tool
        search_tool = Tool(
            name="semantic_search",
            func=self._semantic_search_tool,
            description=(
                "Search for electronic products based on semantic similarity. "
                "Input should be a search query describing what kind of product you're looking for. "
                "The tool will return relevant products with their details including name, category, "
                "description, use cases, and technical specifications."
            )
        )
        
        tools = [search_tool]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful shopping assistant that helps users find electronic products. "
                "You have access to a semantic search tool that can find products based on their "
                "descriptions, features, and use cases. When a user asks about products, use the "
                "semantic_search tool to find relevant items, then provide a helpful and informative "
                "response based on the search results. Always search first before answering questions "
                "about products. Format your responses in a clear, friendly manner and highlight "
                "the most relevant aspects based on what the user is looking for." 
                "Always suggest an alternative if you find any that somehow fit the user's intent."
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def search_and_respond(self, user_query: str) -> str:
        """Process user query and return LLM-generated response."""
        try:
            response = self.agent_executor.invoke({"input": user_query})
            return response["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"