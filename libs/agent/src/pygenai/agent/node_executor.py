"""Agent node executor. This module encapsulates the business logic and execution nodes for the agent, separating it from the builder to allow for clean modifications or extensions of reasoning behavior without cluttering the infrastructure setup."""
from __future__ import annotations

import uuid
from typing import Any, TypeAlias

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)

from pygenai.agent.utils.mcp_manager import get_mcp_manager
from pygenai.core.base_memory import MemoryTypes
from pygenai.core.base_retriever import BaseRetriever
from pygenai.core.logger import get_logger
from pygenai.agent.models import AgentState

from jinja2 import Environment, PackageLoader, select_autoescape

# Configuración del entorno de Jinja para buscar en el paquete pygenai.agent.prompts
env = Environment(
    loader=PackageLoader("pygenai.agent", "prompts"),
    autoescape=select_autoescape()
)

logger = get_logger(__name__)
MCPConnection: TypeAlias = StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection


class AgentNodeExecutor:
    """Encapsulates the business logic and execution nodes for the agent.

    Separating this from the builder allows modifying or extending reasoning
    behavior cleanly without cluttering the infrastructure setup.
    """

    def __init__(self, llm: ChatLiteLLM, rag_sources: dict[str, BaseRetriever], memory: MemoryTypes | None,
                 skills: list[Any], mcps: dict[str, MCPConnection]) -> None:
        """Initializes the AgentNodeExecutor with the provided LLM, RAG sources, memory, skills, and MCP connections."""
        self._llm = llm
        self._rag_sources = rag_sources
        self._memory = memory
        self._skills = skills
        self._mcps = mcps
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_manager = get_mcp_manager()

    async def _start_mcp_client(self) -> None:
        """Starts the MultiServerMCPClient if MCP servers are configured and the client is not already initialized."""
        if self._mcps and not self._mcp_client:
            self._mcp_client = MultiServerMCPClient(self._mcps)  # type: ignore[arg-type]
            self._mcp_manager.current_config_hash = self._mcp_manager._get_hash(self._mcps)

    async def prepare_rag_query_node(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Prepares the RAG query and retrieves relevant texts if RAG sources are configured."""
        configurable = config.get("configurable", {})
        session_id = configurable.get("thread_id") or str(uuid.uuid1())
        if not self._rag_sources:
            return {"session_id": session_id}

        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {"session_id": session_id}

        last_msg = messages[-1]
        rag_query = last_msg.content if isinstance(last_msg, HumanMessage) \
            else getattr(last_msg, "content", str(last_msg))
        rag_texts = []

        template = env.get_template("node_rag.jinga")
        decision_prompt = template.render(rag_query=rag_query)

        try:
            decision_response = await self._llm.ainvoke([HumanMessage(content=decision_prompt)])
            decision_text = str(decision_response.content).strip().upper()
            if "YES" not in decision_text:
                return {"session_id": session_id}
            _, question_ = decision_text.split("YES: ", 1)
        except Exception:
            question_ = ""

        for rag_name, rag_retriever in self._rag_sources.items():
            try:
                retrieved_content = rag_retriever.retrieve_and_format(question_)
                rag_texts.append(f"--- Content from RAG {rag_name} [{retrieved_content}]")
            except Exception as e:
                rag_texts.append(f"--- RAG [{rag_name}] error loading information: {e} ---")

        return {"rag_query": question_, "rag_texts": rag_texts, "session_id": session_id}

    async def prepare_tools_node(self, state: AgentState) -> dict[str, Any]:
        """Prepares the list of available tools, including those from MCP servers if configured."""
        available_tools = list(self._skills)
        if self._mcps:
            try:
                await self._start_mcp_client()
                mcp_tools = await self._mcp_manager.get_active_tools(self._mcps, self._mcp_client)
                available_tools.extend(mcp_tools)
            except Exception as e:
                logger.warning(f"[Warning: Failed to load tools from MCP servers: {e}]")

        return {"tools": [t.name for t in available_tools]}

    async def memory_node(self, state: AgentState) -> dict[str, Any]:
        """Memory node that retrieves messages from memory if a memory system is configured."""
        current_context = state.get("context", {})
        if self._memory:
            try:
                session_id = state.get("session_id", "")
                current_context = await self._memory.get_messages(session_id)
            except Exception as e:
                current_context["memory_error"] = str(e)
        return {"context": current_context, "memory_data": current_context}

    async def reasoning_node(self, state: AgentState) -> dict[str, Any]:
        """Reasoning node that processes messages, RAG texts, and tools to generate a response."""
        messages = list(state.get("messages", []))
        rag_texts = state.get("rag_texts", [])
        tools = await self._mcp_manager.get_active_tools(self._mcps, self._mcp_client) if self._mcps else []
        current_context = state.get("context", {})
        memory_data = state.get("memory_data", None)

        template = env.get_template("node_reasoning.jinga")
        system_prompt = template.render(
            rag_texts=rag_texts,
            memory_data=memory_data
        ).strip()

        react_agent = create_agent(model=self._llm, tools=tools, system_prompt=system_prompt)
        agent_response = await react_agent.ainvoke({"messages": messages})
        agent_messages = agent_response.get("messages", [])
        updated_messages = current_context + [msg for msg in agent_messages if msg not in messages]

        return {"messages": updated_messages, "context": current_context}

    async def save_memory_node(self, state: Any) -> dict[str, Any]:
        """Saves the current messages to memory if a memory system is configured."""
        if self._memory:
            session_id = state.get("session_id", "")
            user_messages = state.get("messages", [])

            await self._memory.update_messages(session_id, user_messages)

            if len(user_messages) > 6:
                existing_summary = ""
                try:
                    existing_summary = await self._memory.get_summary(session_id)
                except Exception:
                    logger.warning(f"Memory summary not found for session {session_id}")
                    existing_summary = None

                template = env.get_template("node_memory.jinga")
                summary_prompt = template.render(
                    existing_summary=existing_summary,
                    user_messages=user_messages
                )

                try:
                    summary_response = await self._llm.ainvoke([HumanMessage(content=summary_prompt)])
                    updated_summary = summary_response.content.strip()

                    if updated_summary:
                        await self._memory.add_summary(session_id, updated_summary)
                except Exception:
                    logger.warning("Failed to update memory summary.", exc_info=True)

        return {}

    async def close(self) -> None:
        """Closes any open connections, such as MCP clients, to free resources."""
        if self._mcp_client and hasattr(self._mcp_client, "close"):
            await self._mcp_client.close()
