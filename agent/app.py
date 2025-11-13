import asyncio
import json
import os

from mcp import Resource
from mcp.types import Prompt

from agent.mcp_client import MCPClient
from agent.openai_client import OpenAIClient
from agent.models.message import Message, Role
from agent.prompts import SYSTEM_PROMPT


# https://remote.mcpservers.org/fetch/mcp
# Pay attention that `fetch` doesn't have resources and prompts

async def main():
    #TODO:
    # 1. Create MCP client and open connection to the MCP server (use `async with {YOUR_MCP_CLIENT} as mcp_client`),
    #    mcp_server_url="http://localhost:8005/mcp"
    # 2. Get Available MCP Resources and print them
    # 3. Get Available MCP Tools, assign to `tools` variable, print tool as well
    # 4. Create OpenAIClient
    # 5. Create list with messages and add there SYSTEM_PROMPT with instructions to LLM
    # 6. Add to messages Prompts from MCP server as User messages
    # 7. Create console chat (infinite loop + ability to exit from chat + preserve message history after the call to OpenAIClient client)
    async with MCPClient('http://localhost:8005/mcp') as mcp_client:
        resources = await mcp_client.get_resources()
        print(f'Available MCP resources: \n{resources}')
        tools = await mcp_client.get_tools()
        openai_client = OpenAIClient(os.getenv('OPENAI_API_KEY'), 'gpt-5-nano', tools, mcp_client)
        messages: list[Message] = [Message(role = Role.SYSTEM, content = SYSTEM_PROMPT)]

        mcp_prompts: list[Prompt]= await mcp_client.get_prompts()
        messages.extend([Message(role=Role.USER, content=prompt.description) for prompt in mcp_prompts])

        while True:
            user_input = input('> ')
            if user_input == 'quit':
                break

            messages.append(Message(role=Role.USER, content=user_input))
            ai_response = await openai_client.get_completion(messages)
            messages.append(ai_response)


if __name__ == "__main__":
    asyncio.run(main())
